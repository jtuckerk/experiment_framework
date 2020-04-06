#! /usr/bin/python
"""
Helper script to download and run experiments from the experiment server.

run ./experiment_runner.py --help to see flags

This expects the experiment server to be running in a directory with the following contents:
config.yaml - an experiment config in yaml. This is downloaded once at the start of this script.
data.zip - any data required by the experiment
scripts.zip - the experiment.py script and the experiment_client.py script
"""

import logging

import os
import subprocess
from subprocess import PIPE
from absl import flags
from absl import app
import torch
import numpy as np
from copy import deepcopy
import importlib
import yaml
from collections import defaultdict

FLAGS = flags.FLAGS
flags.DEFINE_string('config', help='Config file name', default='config.yaml')
flags.DEFINE_string('experiment', help='Experiment module name', default='experiment')
flags.DEFINE_string('address', help='address ip:port of experiment manager server',
                    default='0.0.0.0:8072')
flags.DEFINE_string('server_password', help='password for file server - required for write/modify',
                    default='very_secure')
flags.DEFINE_boolean('download_data', help='download data.zip from experiment directory',
                     default=True)
flags.DEFINE_boolean('download_scripts', help='download scripts.zip from experiment directory',
                     default=True)
flags.DEFINE_boolean('save_models', help='Save and sync all models to the experiment server.',
                     default=False)
flags.DEFINE_boolean('dry_run', help='Run everything but perform no writes. this may repeate experiments.',
                     default=False)
flags.DEFINE_boolean('raise_on_failure', help='Raise the exception from a failing experiment. Otherwise try the next experiment.',
                     default=True)
flags.DEFINE_boolean('consistent_dataset', help='If true init dataset once for all experiments, else re-init for each exp.',
                     default=True)


def DownloadExpSetup(download_data, download_scripts):
  if download_data:
    logging.info('downloading data from %s' % os.environ['SERVER_ADDR'])

    ret = subprocess.run('wget -N ${SERVER_ADDR}/data.zip',
                         stderr=PIPE, env=os.environ, shell=True)
    if ret.returncode != 0:
      logging.warning(ret.stderr.decode())
      raise Exception("Failed data download.")
    else:
      ret = subprocess.run('unzip -o data.zip',
                           stderr=PIPE, env=os.environ, shell=True)
  if download_scripts:
    logging.info('downloading scripts from %s' % os.environ['SERVER_ADDR'])
    ret = subprocess.run('wget -N ${SERVER_ADDR}/scripts.zip',
                         stderr=PIPE, env=os.environ, shell=True)
    if ret.returncode != 0:
      logging.warning(ret.stderr.decode());
      raise Exception("Failed to download scripts.")
    else:
      ret = subprocess.run('unzip -o scripts.zip',
                           stderr=PIPE, env=os.environ, shell=True)
      ret = subprocess.run('touch __init__.py',
                         stderr=PIPE, env=os.environ, shell=True)

class Tracker():
  def __init__(self, console_metrics, exp_info):
    self.tracked_types = defaultdict(lambda: defaultdict(list))
    self.tracked_types['exp_info'] = exp_info
    self.console_metrics = console_metrics
    self.last_logged = {}
    for i in console_metrics:
      self.tracked_types[i] = defaultdict(list)

      self.last_logged[i] = 0

    self.extra = {} # column name: additional_space

    self.columns = ['Log Type']
    self.has_logged = False

  def PopulateData(self, data):
    dd_data = defaultdict(lambda: defaultdict(list), data)
    for k,v in dd_data.items():
      dd_data[k]=defaultdict(list, v)
    self.tracked_types = dd_data

  def AddExpInfo(self, item, value):
    self.tracked_types['exp_info'][item]=value
    
  def Track(self, entry, item, value, add_space=None):
    self.tracked_types[entry][item].append(value)
    if add_space:
      self.extra[item] = add_space

  def _PopulateColumns(self):
    new_columns = False
    for t in self.console_metrics:
      type_dict = self.tracked_types[t]
      for metric_name in type_dict.keys():
        if metric_name not in self.columns:
          new_columns = True
          self.columns.append(metric_name)
    return new_columns

  def _GetStringRep(self, val, size):
    s=str(val)
    s_trunc = s[:size]
    if type(val)==float:
      try:
        exp_ind = s.index('e')
        exp_str = s[exp_ind:]
        s_trunc = s_trunc[:-len(exp_str)]+exp_str
      except Exception:
        pass
    return s_trunc
      

  def _LogSingle(self, log_type, format_str):
    values = [log_type.upper()]
    for v in self.columns[1:]:
      results = self.tracked_types[log_type].get(v, [""])
      if type(results)==list:
        val = results[-1]
      else:
        val = results
      val = self._GetStringRep(val, len(v)+self.extra.get(v,0))
      values.append(val)
    logging.info(format_str.format(*values))

  def Log(self, t):
    new_columns = self._PopulateColumns()
    s = ""
    for i, c in enumerate(self.columns):
      s +='{%d:<%d}' % (i, len(c)+2+self.extra.get(c,0))
    if new_columns or not self.has_logged:
      logging.info(s.format(*self.columns))

    self._LogSingle(t, s)
    self.has_logged = True
        
  def GetMetrics(self):
    return {t: dict(v) for t,v in self.tracked_types.items()}
      
  
def RunExperiment(save_models):
  from experiment_client import ExperimentClient
  # Contrived example of generating a module named as a string
  experiment_module = FLAGS.experiment
  experiment = importlib.import_module(experiment_module)


  EC = ExperimentClient(os.environ['SERVER_ADDR'],
                        dirs_to_sync=['results/', 'models/'],
                        server_password=FLAGS.server_password,
                        dry_run=FLAGS.dry_run)

  # Load dataset once for all experiments from top level configs. Or...
  if FLAGS.consistent_dataset:
    dataset = experiment.InitDataset(EC.config)

  logging.info("%s experiments in config." % len(EC.configs))
  while EC.MoreExperiments():

    h = EC.GetExperiment()
    if h:
      try:
        # ...Or use dataset configs with distinct hyperparams for each experiment.
        # can add randomization/data augmentation to a single loaded dataset in run_one as well
        if not FLAGS.consistent_dataset:
          dataset = experiment.InitDataset(h['hyperparameters'])

        mt = Tracker(['train', 'val'], exp_info=h)
        model = experiment.CreateModel(h['hyperparameters'])

        ckpt = h['hyperparameters'].get('model_checkpoint', None)
        if ckpt:
          experiment.LoadCheckpoint(model, ckpt)

        model = experiment.RunOne(h['hyperparameters'], model, dataset, mt)
        EC.SaveResults(mt.GetMetrics(), h['experiment_hash'])
        if save_models:
          EC.SaveModel(experiment.SaveModel, model, h['experiment_hash'])
        del model
        torch.cuda.empty_cache()
      except Exception as e:
        logging.warning('Failed to run or save exp %s due to %s\n%s' % (
          h['experiment_hash'], e, h))
        EC.MarkIncomplete(h['experiment_hash'])
        if FLAGS.raise_on_failure:
          raise
      except KeyboardInterrupt:
        logging.warning('Exited experiment  %s\n%s' % (
          h['experiment_hash'], h))
        EC.MarkIncomplete(h)
        raise


def main(argv):
  from imp import reload
  reload(logging)
  level = logging.getLevelName("INFO")
  logging.basicConfig(
    level=level,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
  )

  os.environ['SERVER_ADDR']=FLAGS.address

  DownloadExpSetup(FLAGS.download_data, FLAGS.download_scripts)
  RunExperiment(FLAGS.save_models)

if __name__ == '__main__':
   app.run(main)

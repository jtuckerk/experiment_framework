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

# imports the torch_xla package
import torch_xla
import torch_xla.core.xla_model as xm

FLAGS = flags.FLAGS
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

def RunExperiment(save_models):
  from experiment_client import ExperimentClient
  from experiment import init_dataset, run_one

  dev = xm.xla_device()
  EC = ExperimentClient(os.environ['SERVER_ADDR'],
                        dirs_to_sync=['results/', 'models/'],
                        server_password=FLAGS.server_password,
                        dry_run=FLAGS.dry_run)

  # Load dataset once for all experiments from top level configs
  # TODO setup to load data once and set hyperparams (e.g. data aug) on each exp
  if FLAGS.consistent_dataset:
    dataset = init_dataset(EC.config, dev=dev)

  logging.info("%s experiments in config." % len(EC.configs))
  while EC.MoreExperiments():

    h = EC.GetExperiment()
    if h:
      try:
        if not FLAGS.consistent_dataset:
          dataset = init_dataset(h['hyperparameters'], dev=dev)

        results, model = run_one(h['hyperparameters'], dataset, dev=dev)
        EC.SaveResults(h, results)
        if save_models:
          EC.SaveModel(h, model)
        del model
        torch.cuda.empty_cache()
      except Exception as e:
        logging.warning('Failed to run or save exp %s due to %s\n%s' % (
          h['experiment_hash'], e, h))
        EC.MarkIncomplete(h)
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

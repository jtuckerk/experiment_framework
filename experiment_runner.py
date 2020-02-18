#! /usr/bin/python
"""
Helper script to download and run experiments from the experiment server.

run ./experiment_runner.py --help to see flags

This expects the experiment server to be running in a directory with the following contents:
config.yaml - an experiment config in yaml. This is downloaded once at the start of this script.
data.zip - any data required by the experiment
scripts.zip - the experiment.py script and the experiment_client.py script
"""


import os
import subprocess
from subprocess import PIPE
import logging
from absl import flags
from absl import app
import torch

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
flags.DEFINE_boolean('raise_on_failure', help='Raise the exception from a failing experiment. Otherwise try the next experiment.',
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
      raise Exception("Failed data download.")
    else:
      ret = subprocess.run('unzip -o scripts.zip',
                           stderr=PIPE, env=os.environ, shell=True)
      ret = subprocess.run('touch __init__.py',
                         stderr=PIPE, env=os.environ, shell=True)

def RunExperiment(save_models):
  from experiment_client import ExperimentClient
  from experiment import init_dataset, run_one
  
  EC = ExperimentClient(os.environ['SERVER_ADDR'],
                        dirs_to_sync=['results/', 'models/'],
                        server_password=FLAGS.server_password)  
  
  dataset = init_dataset(EC.config)

  logging.info("%s experiments in config." % len(EC.configs))
  while EC.MoreExperiments():

    h = EC.GetExperiment()
    if h:
      try: 
        train_results, val_results, model = run_one(h['hyperparameters'], dataset)
        EC.SaveResults(h, train_results, val_results)
        if save_models:
          EC.SaveModel(h, model)
        del model
        torch.cuda.empty_cache()
      except Exception as e:
        logging.warning('Failed to run or save exp %s due to %s\n%s' % (
          h['experiment_hash'], e, h))
        if FLAGS.raise_on_failure:
          raise
      

def main(argv):
  logger = logging.getLogger()
  level = logging.getLevelName("INFO")
  logger.setLevel(level)

  os.environ['SERVER_ADDR']=FLAGS.address

  DownloadExpSetup(FLAGS.download_data, FLAGS.download_scripts)
  RunExperiment(FLAGS.save_models)
    
if __name__ == '__main__':
   app.run(main)
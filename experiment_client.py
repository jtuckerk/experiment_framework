# experiment client

from absl import flags
import logging
import time
from yaml import safe_load, dump
import http.client
import itertools
import hashlib
import os
from lxml import html
import time

import torch

logger = logging.getLogger()
level = logging.getLevelName("INFO")
logger.setLevel(level)

HASH_SIZE = 9
LOCK_FILE = '.lockfile'
MAX_LOCK_ATTEMPS = 5
RESULTS_DIR = './results/'
MODEL_DIR = './models/'
STARTED_DIR = './.started/'

"""
Dir structure
- experiment_name
  - config.yaml
  - .started/
    - a3b42ed
  - .lockfile
  - results/
    - a3b42ed
"""

class HttpClient():

  def __init__(self, address, password):
      self.conn = http.client.HTTPConnection(address)
      self.password_headers = {'magicpass': password}
      
  def GetFile(self, filename, overwite=False):
    assert not filename.endswith('/')
    self.conn.request("GET", filename, headers=self.password_headers)
    resp = self.conn.getresponse()
    logging.debug("GET %s status: %s %s" % (filename, resp.status, resp.reason))
    if resp.status != 200:
      return False
    try: 
      with open(filename, 'wb') as f:
        f.write(resp.read())
      return True
    except Exception as e:
      logging.warning("failed to write %s, it probably already exists. %s" % (filename, e))
    
  def ListDir(self, dirname):
    assert dirname.endswith('/')
    self.conn.request("GET", dirname, headers=self.password_headers)
    resp = self.conn.getresponse()
    logging.debug("GET %s status: %s %s" % (dirname, resp.status, resp.reason))
    if resp.status != 200:
      return []
    try: 
      root = html.fromstring(resp.read())
      # directories end with /
      directories_and_files = root.xpath('//body//ul//li//text()')
      files = [x for x in directories_and_files if not x.endswith('/')]
      return files
    except Exception as e:
      logging.warning("failed to parse %s list response. %s" % (dirname, e))
      raise

  def PutFile(self, filename, overwrite=False, contents=None):
    mode = 'rb'
    if not contents: 
      with open(filename, mode) as f:
        file_contents = f.read()
    else:
      file_contents = contents
    assert not filename.endswith('/')
    headers = self.password_headers.copy()
    if overwrite:
      headers['overwrite'] = 'true'
    remote_path = os.path.join('./', filename)
    self.conn.request("PUT", remote_path, file_contents, headers=headers)
    resp = self.conn.getresponse()
    logging.debug("GET %s status: %s %s" % (filename, resp.status, resp.reason))
    return resp.status == 201 or resp.status == 200

  def DeleteFile(self, filename):
    assert not filename.endswith('/')
    self.conn.request("DELETE", filename, headers=self.password_headers)
    resp = self.conn.getresponse()

    if resp.status == 200:
      return True
    else:
      return False
    
class FileSyncer():
  def __init__(self, http_client, dirs_to_sync):
    self.http_client = http_client

    self.dirs_to_sync = dirs_to_sync
        
  def CheckSyncStatus(self):
    files_missing_remote = []  
    for d in self.dirs_to_sync:
      remote_files = set(self.http_client.ListDir(d))
      local_files = set(os.listdir(d))
      files_missing_remote += [d + x for x in list(local_files - remote_files)]
    return files_missing_remote

  def SyncMissingFiles(self):
    missing_files = self.CheckSyncStatus()
    for f in missing_files:
      self.http_client.PutFile(f)
      logging.info("Syncing %s" % f)
  
class ExperimentClient():
  def __init__(self, address, dirs_to_sync=[], server_password="very_secure"):
    self.holding_lock = False

    for d in dirs_to_sync:
      if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
      
    self.http_client = HttpClient(address, server_password)
    self.file_syncer = FileSyncer(self.http_client, dirs_to_sync)
    self.file_syncer.SyncMissingFiles()
    config_file = 'config.yaml'
    assert self.http_client.GetFile(config_file), "could not get config file for experiment"



    self.config = self.LoadConfig(config_file)
    self.configs = self.InitializeExperiments()

    self.n_experiments_left = len(self.configs)


  def __del__(self):
    if self.holding_lock:
      self.ReleaseLock()
      
  def StartedExperiments(self):
    return self.http_client.ListDir(STARTED_DIR)


  def GetLock(self):
    for i in range(MAX_LOCK_ATTEMPS):
      if self.http_client.PutFile(LOCK_FILE, contents=b'locked'):
        self.holding_lock = True
        return True
      time.sleep(1)
    raise Exception("Failed to grab lock")
  
  def ReleaseLock(self):
    assert self.http_client.DeleteFile(LOCK_FILE), "Bad State trying to release a lock file failed"
    self.holding_lock = False
    logging.debug("Released lock")

  def DoLocked(self, fn):
    self.GetLock()
    r_val = fn()
    self.ReleaseLock()
    return r_val
  
  def LoadConfig(self, filename):
    with open(filename, 'r') as f:
      config_dict = safe_load(f.read())

    hyperparams = config_dict['hyperparameters']
    for h, val in hyperparams.items():
      if type(val) != list:
        hyperparams[h] = [val]
    logging.debug("Config Loaded")
    logging.debug(config_dict)
    return config_dict

  def GetExperimentConfigs(self, config_dict):
    hyperparams = config_dict['hyperparameters']
    del config_dict['hyperparameters']

    # get all experiments
    params_lists = []
    for k in sorted(hyperparams.keys()):
      params_lists.append(hyperparams[k])

    experiment_params_lists = []
    for element in itertools.product(*params_lists):
      experiment_params_lists.append(element)

    # unpack hyperparams
    hyperparam_dicts = []
    for exp in experiment_params_lists:
      h = {}
      for i, k in enumerate(sorted(hyperparams.keys())):
        h[k] = exp[i]
      hyperparam_dicts.append(h)

    for h in hyperparam_dicts:
      new_dict = config_dict.copy()
      new_dict['hyperparameters'] = h
      yield new_dict

  def GetExperimentHash(self, config_dict):
    return hashlib.md5(str(config_dict).encode('utf-8')).hexdigest()[:HASH_SIZE]

  def InitializeExperiments(self):
    configs = {}
    for c in self.GetExperimentConfigs(self.config):
      exp_hash = self.GetExperimentHash(c)
      c['experiment_hash'] = exp_hash
      configs[exp_hash] = c
    return configs
    
  def _GetExperiment(self):
    completed_experiments = self.StartedExperiments()
    all_experiments = set(self.configs.keys())

    experiments_left = all_experiments - set(completed_experiments)
    self.n_experiments_left = len(experiments_left)
    logging.debug("all experiments: %s" % all_experiments)
    logging.debug("complete experiments: %s" % completed_experiments)
    logging.info("%d experiments left." % self.n_experiments_left)
    if not experiments_left:
      logging.debug('no experiments left')
      return

    next_exp = experiments_left.pop()
    logging.debug("found exp to run: %s" % next_exp)
    self.http_client.PutFile(os.path.join(STARTED_DIR, next_exp), contents=b'started')
    return next_exp
                          
  def GetExperiment(self):
    exp = self.DoLocked(self._GetExperiment)
    return self.configs.get(exp, None)

  def MoreExperiments(self):
    return self.n_experiments_left >= 1

  def SaveResults(self, h, train_results, val_results):
    filename = os.path.join(RESULTS_DIR, h['experiment_hash'])
    if not os.path.exists(filename): 
      with open(filename, 'x') as f:
        out = {'train_results': train_results,
               'val_results': val_results,
               'exp_info': h}
        f.write(dump(out))
    else:
      logging.warning("Tried overwriting existing result")
    self.file_syncer.SyncMissingFiles()

  def SaveModel(self, h, model):
    filename = os.path.join(MODEL_DIR, h['experiment_hash'])
    if os.path.exists(filename): 
      logging.warning("saved model already exists.")
      filename+='.new'
    torch.save(model.state_dict(), filename)      
    self.file_syncer.SyncMissingFiles()
    
def _TestExperiment(hash_exp):
  for i in range(10):
    logging.warning("exp %s run %s" % (hash_exp['experiment_hash'], i ))
    time.sleep(1)
  logging.warning("exp done %s" % hash_exp)


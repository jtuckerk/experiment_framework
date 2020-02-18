## Experiment Framework

The files in this repo make up a framework for running experiments over grids of hyperparameters on multiple machines.
<br><br>
A single machine (usually my laptop) runs the server from an experiment directory which contains all of the relevant experiment scripts and data.

### The key framework components:

* Server - file_server.py
  * A simple http server that acts as the Experiment Server
* Client - experiment_client.py
  * Handles Running experiments and syncing results and models to the server.
* Experiment Runner - experiment_runner.py
  * A helper for downloading all the scripts and data needed to run an experiment on a remote machine.

### User defined components

* experiment.py
  * Defines ```init_dataset``` and ```run_one``` which each take a set of hyperparameters from the ExperimentClient.
* data.zip
  * Any files needed by the ```init_dataset``` function in experiment.py
* scripts.zip
  * experiment_client.py
  * experiment.py


### Usage
Run the file server in your experiment directory:

```
./file_server.py --port=8030
```

Download the experiment runner on some remote machines and run the experiments.
```
wget  ${SERVER_ADDR}/experiment_runner.py;
python ./experiment_runner.py \
  --save_models=False \
  --download_scripts \
  --download_data=False \
  --address=${SERVER_ADDR} \
  --server_password="very_secure";
```

See ```experiment.py``` for an example of an experiment and ```config.yaml``` for an experiment config example.

### Example config
Define an experiment with lists of hyperparameters to be used in an experiment grid. The example below would spawn 8 experiments.

```
experiment_name: sentiment_analysis1
dataset: 'imdb.pt'
hyperparameters:
  batch_size:
    -  128
    -  64
  learning_rate:
    - .03
    - .1
  loss_fn:
    - 'xent'
  epochs:
    - 10
  fc_sizes:
    - [2048]
    - [4096]
  vocab_size:
    - 70
  char_embedding_size:
    -  8
  activation:
    -  'relu'
```

This uses Pytorch for saving and loading models. Eventually, the client will be framework agnostic with save and load callbacks, but this was good enough for now.

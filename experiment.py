def init_dataset(exp_info):
  # do whatever data loading using the values defined in the top level (not hyperparameters) of the config.
  # this setup is optimized for a loading a dataset into memory once and then running many models.
  return dataset

def run_one(h, dataset):
  # takes a single set of hyperparameters h and the dataset returned from init_dataset
  return results, model

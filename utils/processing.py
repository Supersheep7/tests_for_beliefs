import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.cfg import load_cfg
from transformer_lens import HookedTransformer
import torch as t
from datasets import load_dataset
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
cfg = load_cfg()

class TrueFalseBuilder():
  def __init__(self, clean=True):
    self.path = f'{ROOT}/data/datasets/true_false'
    self.clean = clean

  def get_dataset(self):
    dfs = {}
    df_all = pd.DataFrame()
    to_exclude = ['geonames.csv', 'common_claim.csv', 'likely_old.csv']
    for file in tqdm(os.listdir(self.path), desc="Processing files"):
      if file.endswith('.csv') and file not in to_exclude:
        df = pd.read_csv(os.path.join(self.path, file))
        if self.clean:
          # Drop columns
          if file in ['cities_cities_disj.csv', 'cities_cities_conj.csv']:
            df.drop(columns=['city1', 'city2', 'country1', 'country2', 'correct_country1', 'correct_country2', 'statement1', 'label1', 'statement2', 'label2'], inplace=True)
          elif file in ['cities.csv', 'neg_cities.csv']:
            df.drop(columns=['city', 'country', 'correct_country'], inplace=True)
          elif file in ['larger_than.csv', 'smaller_than.csv']:
            df.drop(columns=['n1', 'n2', 'diff', 'abs_diff'], inplace=True)
          elif file == 'counterfact_true_false.csv':
            df.drop(columns=['relation', 'subject', 'target', 'true_target'], inplace=True)
          elif file == 'likely.csv':
            df.drop(columns=['likelihood'], inplace=True)
        df['filename'] = file
        dfs[file] = df
        df_all = pd.concat([df_all, df])

    print("===================")
    print("WATCH OUT! Datapoints have different column entries depending on the csv")
    print()
    return dfs, df_all

  def debug(self):
    print(os.listdir(self.path))

def get_data():
    databuilder = TrueFalseBuilder()
    dfs, df_all = databuilder.get_dataset()
    # Preprocessing
    df_all = df_all[df_all['filename'] != 'likely.csv']

    # Trim the counterfact_true_false dataset to have balanced classes and a max of 1000 samples per class
    target_df = df_all[df_all['filename'] == 'counterfact_true_false.csv']
    label_0 = target_df[target_df['label'] == 0]
    label_1 = target_df[target_df['label'] == 1]
    retain_per_label = 1000
    sampled_0 = label_0.sample(n=retain_per_label, random_state=cfg["common"]["seed"])
    sampled_1 = label_1.sample(n=retain_per_label, random_state=cfg["common"]["seed"])
    sampled_target_df = pd.concat([sampled_0, sampled_1])
    df_all = pd.concat([df_all[df_all['filename'] != 'counterfact_true_false.csv'], sampled_target_df])
    # df_all['statement'] = df_all['statement'] + ' This sentence is:'

    df_train, df_test = train_test_split(
        df_all,
        test_size=0.5,
        stratify=df_all['filename']
    )

    df_trimmed = df_train.iloc[:-(len(df_train) % cfg["tlens"]["batch_extractor"]), :]
    x = list(df_trimmed['statement'])
    y = list(df_trimmed['label'])
    return x, y

def get_activations(model: HookedTransformer, data, modality: str = 'residual'):
    return None
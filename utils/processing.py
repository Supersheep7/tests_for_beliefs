import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.cfg import load_cfg
from transformer_lens import HookedTransformer
from transformer_lens.utils import get_act_name
from typing import List, Tuple
from jaxtyping import Float, Int
import torch as t
from datasets import load_dataset
import pandas as pd
import os
import gc 
from tqdm import tqdm
import random
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
cfg = load_cfg()

class CoherenceBuilder():
  def __init__(self, clean=True):
    self.path = f'{ROOT}/data/datasets/coherence'
    self.batch_extractor = cfg["tlens"]["batch_extractor"]
    with open(os.path.join(ROOT, 'data/datasets/coherence/curated_dataset.pkl'), 'rb') as f:
      self.curated_dataset = pickle.load(f)

  def get_data_split(self, task: str, other_dataset=None, cutoff=1500):

    if task in ['negation', 'disjunction', 'conjunction']:
        # In this case curated dataset is the logical dataset + the remainder
        remainder = curated_dataset[~curated_dataset['filename'].isin(['common_claim_true_false.csv', 'companies_true_false.csv', 'counterfact_true_false.csv'])]
        curated_dataset = pd.concat([remainder, other_dataset])
        train_set, test_set = train_test_split(curated_dataset, test_size=0.2, random_state=42, stratify=curated_dataset['filename'])
        test_df = test_set.dropna()
        
    elif task == 'entailment':
        remainder = curated_dataset[~curated_dataset['filename'].isin(['common_claim_true_false.csv', 'companies_true_false.csv', 'counterfact_true_false.csv',
                                                                      'cities.csv', 'cities_cities_conj.csv', 'cities_cities_disj.csv', 'neg_cities.csv'
                                                                      ])]
        curated_dataset = pd.concat([remainder, other_dataset])
        train_set, test_set = train_test_split(curated_dataset, test_size=0.2, random_state=42, stratify=curated_dataset['filename'])
        test_df = test_set.dropna()        

    # Trim for batch size
    train_set = train_set.iloc[:-(len(train_set) % self.batch_extractor), :]
    X_clean_train = list(train_set['statement'])
    y_clean_train = list(train_set['label'])
    
    return (X_clean_train, y_clean_train, test_df)

  def get_neg_data(self):
    with open(os.path.join(self.path, 'neg_dataset.pkl'), 'r') as f:
       raw = pickle.load(f)
    X_train, y_train, test_data  = self.get_data_split('negation', other_dataset=raw)
    data_pos = test_data['statement'].tolist()   
    data_neg = test_data['new_statement'].tolist()   
    return (X_train, y_train), data_pos, data_neg
  
  def get_or_data(self):
    test_data = pd.read_csv(os.path.join(self.path, 'disjunction.csv'))
    data_atom = test_data['statement'].tolist()       
    data_or = test_data['new_statement'].tolist()
    return data_atom, data_or

  def get_and_data(self):
    test_data = pd.read_csv(os.path.join(self.path, 'conjunction.csv'))
    data_atom = test_data['statement'].tolist()   
    data_and = test_data['new_statement'].tolist()
    return data_atom, data_and

  def get_ifthen_data(self):
    test_data = pd.read_csv(os.path.join(self.path, 'entailment.csv'))
    data_atom = test_data['statement'].tolist()       
    data_and = test_data['new_statement'].tolist()         
    data_ifthen = test_data['hop_statement'].tolist() 
    return data_atom, data_and, data_ifthen

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
    print("True_False Dataset Accessed")
    print("===================")
    print("DF head:", df_all.head())
    print()
    return dfs, df_all

  def debug(self):
    print(os.listdir(self.path))

def to_split(df, domains):
   return df[df['filename'] == domains[0]] if len(domains) == 1 else df[df['filename'].isin(domains)]

def stratify(df, category_column='filename', test_size=0.2, random_state=42):
   
    # Get unique categories
    categories = df[category_column].unique()
    
    # Split categories themselves (not individual rows)
    train_cats, test_cats = train_test_split(categories, test_size=test_size, random_state=random_state)
    
    # Assign rows based on category split
    train_df = df[df[category_column].isin(train_cats)].reset_index(drop=True)
    test_df = df[df[category_column].isin(test_cats)].reset_index(drop=True)
    
    return train_df, test_df

def split_curated_df_logic(df):

   train_df, test_df = stratify(df)

   train_datas = [
                train_df,
                to_split(train_df, ['common_claim_true_false.csv']),
                to_split(train_df, ['common_claim_true_false.csv', 'conj_common_claim_true_false.csv', 'disj_common_claim_true_false.csv', 'neg_common_claim_true_false.csv']),
                to_split(train_df, ['common_claim_true_false.csv', 'disj_common_claim_true_false.csv', 'conj_common_claim_true_false.csv']),
                to_split(train_df, ['common_claim_true_false.csv', 'conj_common_claim_true_false.csv', 'neg_common_claim_true_false.csv']),
                to_split(train_df, ['common_claim_true_false.csv', 'disj_common_claim_true_false.csv', 'neg_common_claim_true_false.csv']),
                to_split(train_df, ['common_claim_true_false.csv', 'conj_common_claim_true_false.csv']),
                to_split(train_df, ['common_claim_true_false.csv', 'disj_common_claim_true_false.csv']),
                to_split(train_df, ['common_claim_true_false.csv', 'neg_common_claim_true_false.csv']),
                to_split(train_df, ['conj_common_claim_true_false.csv']),
                to_split(train_df, ['disj_common_claim_true_false.csv']),
                to_split(train_df, ['neg_common_claim_true_false.csv']),
   ]

   test_datas = [
                 to_split(test_df, ['cities.csv']), 
                 to_split(test_df, ['neg_cities.csv']), 
                 to_split(test_df, ['cities_cities_conj.csv']),
                 to_split(test_df, ['cities_cities_disj.csv']),
                 to_split(test_df, ['common_claim_true_false.csv']),
                 to_split(test_df, ['conj_common_claim_true_false.csv']),
                 to_split(test_df, ['disj_common_claim_true_false.csv']),
                 to_split(test_df, ['neg_common_claim_true_false.csv']),
                 to_split(test_df, ['companies_true_false.csv']),
                 to_split(test_df, ['sp_en_trans.csv']),
                 to_split(test_df, ['neg_sp_en_trans.csv']),
                 to_split(test_df, ['larger_than.csv']),
                 to_split(test_df, ['smaller_than.csv']),
                 to_split(test_df, ['counterfact_true_false.csv'])
                ]

   return (train_datas, test_datas)

def split_curated_df_domains(df):

  train_df, test_df = stratify(df)

  train_datas = [
              train_df,
              to_split(train_df, ['common_claim_true_false.csv']),
              to_split(train_df, ['common_claim_true_false.csv', 'cities.csv']),
              to_split(train_df, ['cities.csv']),
              to_split(train_df, ['common_claim_true_false.csv', 'companies_true_false.csv']),
              to_split(train_df, ['companies_true_false.csv']),
              to_split(train_df, ['common_claim_true_false.csv', 'sp_en_trans.csv']),
              to_split(train_df, ['sp_en_trans.csv']),
              to_split(train_df, ['common_claim_true_false.csv', 'larger_than.csv']),
              to_split(train_df, ['larger_than.csv']),
              to_split(train_df, ['common_claim_true_false.csv', 'counterfact_true_false.csv']),
              to_split(train_df, ['counterfact_true_false.csv'])
  ]

  test_datas = [
              to_split(test_df, ['cities.csv']), 
              to_split(test_df, ['common_claim_true_false.csv']),
              to_split(test_df, ['companies_true_false.csv']),
              to_split(test_df, ['sp_en_trans.csv']),
              to_split(test_df, ['larger_than.csv']),
              to_split(test_df, ['counterfact_true_false.csv'])
            ]
  return (train_datas, test_datas)

def get_data(experiment: str = 'accuracy', sweep: bool = False, logic: str = None):
    print("Getting data...")
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
    df_train, df_test = train_test_split(
          df_all,
          test_size=0.5,
          stratify=df_all['filename']
      )
    if experiment == 'accuracy':
      df_trimmed = df_train.iloc[:-(len(df_train) % cfg["tlens"]["batch_extractor"]), :]
      x = df_trimmed['statement'].astype(str) + " This statement is: "  # We want this like this because we are collecting the directions for the intervention experiment later on
      x = x.tolist()
      y = list(df_trimmed['label'])
      return x, y
    elif experiment == 'visualization':
      df_trimmed = df_all.iloc[:-(len(df_all) % cfg["tlens"]["batch_extractor"]), :]
      x = list(df_trimmed['statement'])
      y = list(df_trimmed['label'])
      return x, y
    elif experiment == 'intervention':
      df_trimmed = df_test.iloc[:-(len(df_test) % cfg["tlens"]["batch_extractor"]), :]
      df_for_int = df_trimmed.copy()
      df_for_int['statement'] = df_trimmed['statement'] + " This statement is: "
      df_true = df_for_int[df_for_int['label'] == 1]
      df_false = df_for_int[df_for_int['label'] == 0]
      if sweep:
        test_sample_true = random.sample(list(df_true['statement']), 100)
        test_sample_false = random.sample(list(df_false['statement']), 100)
        sampled_df_true = df_true[df_true['statement'].isin(test_sample_true)]
        sampled_df_false = df_false[df_false['statement'].isin(test_sample_false)]
        return list(sampled_df_true['statement']), list(sampled_df_true['label']), list(sampled_df_false['statement']), list(sampled_df_false['label'])
      else:
        return list(df_true['statement']), list(df_true['label']), list(df_false['statement']), list(df_false['label'])
    elif experiment == 'coherence':
      databuilder = CoherenceBuilder()
      if logic == 'neg':
        train_df, data_pos, data_neg = databuilder.get_neg_data()
        return train_df, data_pos, data_neg
      elif logic == 'or':
        train_df, data_atom, data_or = databuilder.get_or_data()
        return train_df, data_atom, data_or
      elif logic == 'and':
        train_df, data_atom, data_and = databuilder.get_and_data()
        return train_df, data_atom, data_and
      elif logic == 'ifthen':
        train_df, data_atom, data_and, data_ifthen = databuilder.get_ifthen_data()
        return train_df, data_atom, data_and, data_ifthen
    elif experiment == 'uniformity':
      
      # Get the rest of the data 
      with open(os.path.join(ROOT, 'data/datasets/coherence/neg_dataset.pkl'), 'rb') as f:
        neg_dataset = pickle.load(f)
      with open(os.path.join(ROOT, 'data/datasets/coherence/conj_dataset.pkl'), 'rb') as f:
        conj_dataset = pickle.load(f)
      with open(os.path.join(ROOT, 'data/datasets/coherence/disj_dataset.pkl'), 'rb') as f:
        disj_dataset = pickle.load(f)

      common_neg = neg_dataset[neg_dataset['filename'] == 'neg_common_claim_true_false.csv']
      common_conj = conj_dataset[conj_dataset['filename'] == 'conj_common_claim_true_false.csv']
      common_disj = disj_dataset[disj_dataset['filename'] == 'disj_common_claim_true_false.csv']
      df_all = pd.concat([df_all, common_neg, common_conj, common_disj])

      folds_logic = split_curated_df_logic(df_all)
      folds_domains = split_curated_df_domains(df_all)

      # Stratified uniform split
      
      return folds_logic, folds_domains
    else:
      raise ValueError(f"Unsupported experiment type: {experiment}")

class ActivationExtractor():
    
    def __init__(self, 
                 model: HookedTransformer, 
                 data: List, 
                 labels: List, 
                 device: t.device = t.device('cuda' if t.cuda.is_available() else 'cpu'),
                 half: bool = True,
                 batch_size=32,
                 pos=-1):
        
        self.model = model.to(device)
        self.X = self.batchify(data, batch_size)
        self.y = t.tensor(self.batchify(labels, batch_size), dtype=t.float32).to(device)
        self.hooks = []
        self.activations = {}
        self.half = half
        self.device = device
        self.pos = pos  

    def set_hooks(self, layers, names, attn=False):

        if self.half:
            self.hooks.append(("hook_embed", lambda tensor, hook: tensor.half()))
        
        def get_act_hook(tensor, hook):
            tensor = tensor.detach()
            last_token = tensor[:, self.pos, :, :].unsqueeze(0) if attn else tensor[..., self.pos, :].unsqueeze(0)  
            last_token = last_token.to(dtype=t.float16, device=t.device('cpu'))

            if hook.name in self.activations:
                self.activations[hook.name] = t.cat([self.activations[hook.name], last_token], dim=0)
            else:
                self.activations[hook.name] = last_token

            return tensor

        for layer in layers:
            for name in names:
                self.hooks.append((f"blocks.{layer}.{name}", get_act_hook))
          
                
    def extract_activations_batch(self, 
                                  sentences: t.Tensor, 
                                  model: HookedTransformer, 
                                  ) -> None:
        """
        Extract activations and sets them in self dictionary
        """

        '''sentences.shape == (batch_size 1)'''
        print(f"GPU memory before model.to_tokens: {t.cuda.memory_allocated()/1e9:.2f}GB")
        tokens = model.to_tokens(sentences)
        print(f"GPU memory after model.to_tokens: {t.cuda.memory_allocated()/1e9:.2f}GB")

        '''tokens.shape == (batch_size seq_len)'''

        with t.no_grad():
          with t.amp.autocast(device_type='cuda', dtype=t.float16):
            print(f"GPU memory before model.reset_hooks: {t.cuda.memory_allocated()/1e9:.2f}GB")
            model.reset_hooks()
            print(f"GPU memory after model.reset_hooks: {t.cuda.memory_allocated()/1e9:.2f}GB")
            
            # Forward pass running with hooks
            print(f"GPU memory before model.run_with_hooks: {t.cuda.memory_allocated()/1e9:.2f}GB")
            model.run_with_hooks(
                tokens,
                return_type=None,
                fwd_hooks=self.hooks,
                clear_contexts=True
            )
            print(f"GPU memory after model.run_with_hooks: {t.cuda.memory_allocated()/1e9:.2f}GB")

        return

    def batchify(self, data, batch_size):
        """
        Split data into batches. We need to add padding or something like that
        """
        result = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        assert len(result[-1]) % batch_size == 0, "Data length must be divisible by batch_size"
        return result

    def process(self
    ) -> Tuple[List[Float[t.Tensor, "batch_size n_acts d_model"]], Int[t.Tensor, "batch_size"]]:
        # Process
        for i, batch in enumerate(tqdm(self.X, "Processing")):
            t.cuda.empty_cache()
            print(f"GPU memory before batch {i}: {t.cuda.memory_allocated()/1e9:.2f}GB")
            self.extract_activations_batch(batch, self.model)
            print(f"GPU memory after batch {i}: {t.cuda.memory_allocated()/1e9:.2f}GB")
        
        print("Activation extraction complete.")
        return self.activations, self.y

def get_activations(model: HookedTransformer, data, modality: str = 'residual', focus = None):

    print("Extracting activations...")
    model.to(cfg["common"]["device"])
    model.reset_hooks() 
    statements, labels = data
    extractor = ActivationExtractor(model=model, data=statements, labels=labels, device=cfg["common"]["device"], half=True,
                                      batch_size=cfg["tlens"]["batch_extractor"], pos=-1)
    if modality == 'heads':
        if focus is None:
          extractor.set_hooks([i for i in range(model.cfg.n_layers)],
                              [get_act_name('z')], attn=True)
        else: 
          layer, head = focus
          extractor.set_hooks([layer],
                              [f'blocks.{layer}.attn.hook_z_{head}'], attn=True)
    else:
        if focus is None:
          extractor.set_hooks(
                              [i for i in range(model.cfg.n_layers)],
                              [get_act_name('resid_post')], attn=False)
        else: 
          layer = focus
          extractor.set_hooks(
                              [layer],
                              [get_act_name('resid_post')], attn=False) 
    activations, labels = extractor.process()
    model.to(t.device('cpu'))
    gc.collect()
    t.cuda.empty_cache()
    return activations, labels
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

        tokens = model.to_tokens(sentences)

        '''tokens.shape == (batch_size seq_len)'''

        with t.no_grad():

            model.reset_hooks()
            
            # Forward pass running with hooks
            model.run_with_hooks(
                tokens,
                return_type=None,
                fwd_hooks=self.hooks
            )

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
        for batch in tqdm(self.X, "Processing"):
            self.extract_activations_batch(batch, self.model)
        return self.activations, self.y

def get_activations(model: HookedTransformer, data, modality: str = 'residual'):
    model.to(cfg["common"]["device"])
    model.reset_hooks()
    extractor = ActivationExtractor(model=model, data=data, labels=labels, device=cfg["common"]["device"], half=True,
                                      batch_size=cfg["tlens"]["batch_extractor"], pos=-1)
    if modality == 'heads':
        extractor.set_hooks([i for i in range(model.cfg.n_layers)],
                            [get_act_name('z')], attn=True)
    else:
        extractor.set_hooks(
                            [i for i in range(model.cfg.n_layers)],
                            [get_act_name('resid_post')], attn=False) 
    activations, labels = extractor.process() # Get
    model.to(t.device('cpu'))
    gc.collect()
    t.cuda.empty_cache()
    return activations, labels
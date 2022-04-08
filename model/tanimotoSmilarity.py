import os
import argparse
import torch
import torchvision.transforms as transforms
import pandas as pd
import time
import ray

from model.Model import MSTS
from src.datasets import SmilesDataset
from src.config import input_data_dir, base_file_name, sample_submission_dir, generate_submission_dir, sample_submission_labels_dir, reversed_token_map_dir
from utils import logger, make_directory, load_reversed_token_map, smiles_name_print, str2bool
from rdkit import Chem, DataStructs


predict_file = pd.read_csv(generate_submission_dir)
labels_file = pd.read_csv(sample_submission_labels_dir)
count = 0
sum_Tan = 0
for _, row in predict_file.iterrows(): #Iterate over DataFrame rows as (index, Series) pairs.
    idx = row['file_name']
    smiles_pred = row['SMILES']
    smiles_label = labels_file.loc[_]['SMILES']
    label_idx = labels_file.loc[_]['file_name']
    print("_", _)
    print('pred_idx', idx)
    print('smiles_pred:', smiles_pred)
    print("smiles_label:", smiles_label)
    print('label_idx:', label_idx)
    try:
        ref_pred = Chem.MolFromSmiles(smiles_pred)
        fp_pred = Chem.RDKFingerprint(ref_pred)
    except:
        print('Invalid SMILES:', smiles_pred)

    ref_label = Chem.MolFromSmiles(smiles_label)
    fp_label = Chem.RDKFingerprint(ref_label)

    Tan = DataStructs.TanimotoSimilarity(fp_pred,fp_label)
    print(Tan)
    sum_Tan = sum_Tan + Tan
    count += 1

average = sum_Tan/count
print("average:", average)

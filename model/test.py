import pandas as pd

from src.config import input_data_dir, base_file_name, sample_submission_dir, sample_submission_labels_dir, reversed_token_map_dir
from rdkit import Chem, DataStructs
from Levenshtein import distance as levenshtein_distance


def damerau_levenshtein_distance(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    for i in range(-1,lenstr1+1):
        d[(i,-1)] = i+1
    for j in range(-1,lenstr2+1):
        d[(-1,j)] = j+1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i,j)] = min(
                d[(i-1,j)] + 1, # deletion
                d[(i,j-1)] + 1, # insertion
                d[(i-1,j-1)] + cost, # substitution
            )
            if i and j and s1[i]==s2[j-1] and s1[i-1] == s2[j]:
                d[(i,j)] = min (d[(i,j)], d[i-2,j-2] + cost) # transposition

    return d[lenstr1-1,lenstr2-1]


predict_file = pd.read_csv('good_test.csv')
labels_file = pd.read_csv('/cvhci/temp/zihanchen/data/results_training/E2+2transformer+LSTM_1M_group1/good_test_labels.csv')
count = 0
sum_Tan = 0
sum_Lev = 0
sum_dam = 0
for _, row in predict_file.head(10000).iterrows(): #Iterate over DataFrame rows as (index, Series) pairs.
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
    print("Tanimoto Smililarity:", Tan)
    sum_Tan = sum_Tan + Tan
    count += 1

    #calculate levenshtein distance
    leven = levenshtein_distance(smiles_pred, smiles_label)
    leven = 1 - leven/ max(len(smiles_pred), len(smiles_label))
    print("levenshtein distance:", leven)
    sum_Lev = sum_Lev + leven

    #calculate Damerau-levenshtein distance
    dam_lev = damerau_levenshtein_distance(smiles_pred, smiles_label)
    dam_lev = 1 -dam_lev / max(len(smiles_pred), len(smiles_label))
    print("Damerau-levenshtein distance:", dam_lev)
    sum_dam = sum_dam + dam_lev


average = sum_Tan/count
average_lev = sum_Lev/count
average_dam = sum_dam/count
print("Tanimoto average:", average)
print("levenshtein average:", average_lev)
print("Damerau-levenshtein distance avergae:", average_dam)


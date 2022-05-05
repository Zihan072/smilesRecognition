
'''
Tutorial: Transformer
https://wmathor.com/index.php/archives/1455/
'''

import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data


def get_positional_encoding(max_seq_len, embed_dim):
    # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 最大的序列长度
    positional_encoding = np.array([
        [pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
        if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])

    #if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len): function?

    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # dim 2i 偶数 ::步长为2取元素

    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # dim 2i+1 奇数
    return positional_encoding

positional_encoding = get_positional_encoding(max_seq_len=100, embed_dim=16)
#plt.figure(figsize=(10,10))
#sns.heatmap(positional_encoding)
#plt.title("Sinusoidal Function")
#plt.xlabel("hidden dimension")




# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps
sentences = [
    # enc_input           dec_input         dec_output
    ['ich mochte ein bier P', 'S i want a beer .', 'i want a beer . E'],
    ['ich mochte ein cola P', 'S i want a coke .', 'i want a coke . E']
]

# Padding Should be Zero
src_vocab = {'P' : 0, 'ich' : 1, 'mochte' : 2, 'ein' : 3, 'bier' : 4, 'cola' : 5}
src_vocab_size = len(src_vocab)

tgt_vocab = {'P' : 0, 'i' : 1, 'want' : 2, 'a' : 3, 'beer' : 4, 'coke' : 5, 'S' : 6, 'E' : 7, '.' : 8}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 5 # enc_input max sequence length
tgt_len = 6 # dec_input(=dec_output) max sequence length

def make_data(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]] # [[6, 1, 2, 3, 4, 8], [6, 1, 2, 3, 5, 8]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]] # [[1, 2, 3, 4, 8, 7], [1, 2, 3, 5, 8, 7]]
        # print("enc_input", enc_input)
        # enc_input [[1, 2, 3, 4, 0]]
        # enc_input [[1, 2, 3, 5, 0]]

        enc_inputs.extend(enc_input)
        # print("enc_inputs.extend(enc_input)", enc_inputs)
        # enc_inputs.extend(enc_input) [[1, 2, 3, 4, 0]]
        # enc_inputs.extend(enc_input) [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]

        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_inputs, dec_outputs = make_data(sentences)
# print("enc_inputs", enc_inputs)
# enc_inputs tensor([[1, 2, 3, 4, 0],
#                    [1, 2, 3, 5, 0]])

class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        print("enc_inputs[idx]", enc_inputs[idx])
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

print("MyDataSet(enc_inputs, dec_inputs, dec_outputs)", MyDataSet(enc_inputs, dec_inputs, dec_outputs))
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)

# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048 # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        print("pe", pe.shape)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

model = PositionalEncoding(5, max_len=6)
#print(model)


'''
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
'''

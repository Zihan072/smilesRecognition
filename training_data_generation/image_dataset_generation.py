"""
The code below serves to generate a train image set by inputting dataframes
by group previously generated in the dataframe_generation_by_group.
The images generated are stored in folders generated by sequence length,
and are used for the gernalization of model learning.
"""


import rdkit
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm.auto import tqdm # solve the problem of each iteration of progressbar starts a new line
import click

import warnings
warnings.filterwarnings(action = 'ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# path
# path_all = '/cvhci/temp/zihanchen/data/new_images_1M_group2/' # Saving new data
# if not os.path.exists(path_all):
#     os.mkdir(path_all)
# else:
#     pass
#
#
# path = path_all + '/train' # Saving new image
#
# path_75 = path_all + '/train75+' # Saving new image which length is longer than 75
#
# data_path = '/cvhci/temp/zihanchen/data/train_dataset_3rd_5M/'
# if not os.path.exists(path):
#     os.mkdir(path)
# else:
#     pass
#
# if not os.path.exists(path_75):
#     os.mkdir(path_75)
# else:
#     pass
#
# # Oragainizing the group by the number of core
# # The number of data sameple for one group calculated as
# # the number of total data sample / the number of core
# # ex) 111307682 / 31 = 3700000
# # The number of core can be different by each environment
#
# file_writer = open(path_all + "train.csv", 'w')
# file_writer.write("file_name,SMILES"+"\n")
#
# file_writer_75 = open(path_all + "train75.csv", 'w')
# file_writer_75.write("file_name,SMILES"+"\n")
# #store smiles longer than 75
#
# @click.command()
# @click.option('--group', default=1, help='group number')
#
# def making_data(group):
#     print("group number:", group)
#     count_75 = 0
#     count = 0
#     filtered_df = pd.read_csv(data_path +'/filtered_df_group{}.csv'.format(group))
#     data_len = len(filtered_df)
#     # for idx in tqdm(range(len(filtered_df[filtered_df['group'] == group]))):
#     for idx in range(data_len):
#         # idx += 3700000 * (group-1)
#         # print('idx:',idx, end='\r')
#         # smiles = filtered_df[filtered_df['group'] == group]['SMILES'][idx]
#         smiles = filtered_df['SMILES'][idx]  # this is the representation string
#         if len(smiles) <= 75:
#             count += 1
#             img_name = str(idx) + ".png"
#             #if len(smiles) > 100: continue  # we only choose the samples that are less than 101
#
#
#             smiles_g = Chem.MolFromSmiles(smiles)
#             try:
#                 # smile_plt is the image so we can directly save it.
#                 smile_plt = Draw.MolToImage(smiles_g, size = (300,300))
#
#                 # converting color
#                 # switching r channel with b channel
#                 # im = np.array(smile_plt)
#                 #  r = im[:,:,0]
#                 # g = im[:,:,1]
#                 # b = im[:,:,2]
#                 # convert = np.stack((b,g,r),axis=-1)
#
#
#                 # Making directory by the length of sequnece and saving
#                 # making filename as "the length of smiels"_train_"index"
#                 # ex) 0020_train_4
#                 #dir_name = len(filtered_df['SMILES'][idx])
#                 #dir_name = str(dir_name).zfill(4)
#                 #os.makedirs(os.path.join(path, dir_name), exist_ok=True)
#                 img_full_name = os.path.join(path, img_name)
#                 file_writer.write(img_name + "," + smiles + "\n")
#                 smile_plt.save(img_full_name)  # save the image in png
#                 # np.save(path  + str(dir_name) + '/' +'{0}_train_{1}_{2}.npy'.format(dir_name, group, idx), arr = convert)
#                 assert len(smiles) <= 75
#                 del (smile_plt)
#             except ValueError:
#                 pass
#         else:
#             img_name = str(idx) + ".png"
#             smiles_g = Chem.MolFromSmiles(smiles)
#             print("Index:{0},SMILES:{1} length longer than 75".format(idx, smiles))
#             try:
#                 # smile_plt is the image so we can directly save it.
#                 smile_plt = Draw.MolToImage(smiles_g, size = (300,300))
#                 img_full_name = os.path.join(path_75, img_name)
#                 file_writer_75.write(img_name + "," + smiles + "\n")
#                 smile_plt.save(img_full_name)  # save the image in png
#                 # np.save(path  + str(dir_name) + '/' +'{0}_train_{1}_{2}.npy'.format(dir_name, group, idx), arr = convert)
#                 assert len(smiles) > 75
#                 del (smile_plt)
#             except ValueError:
#                 pass
#
#             count_75 += 1
#         # checking the completion
#         if idx % 1000 == 0 :
#             print('group : {0}, index : {1}'.format(group, idx))
#     print("Number of length >75 is {0}".format(count_75))
#     print("Number of length <=75 is {0}".format(count))
#     del(filtered_df)
#     file_writer.close()

path_all = '/cvhci/temp/zihanchen/data/new_images_5M_100/' # Saving new data
if not os.path.exists(path_all):
    os.mkdir(path_all)
else:
    pass

data_path = '/cvhci/temp/zihanchen/data/train_dataset_20M'


path = path_all + 'train' # Saving new image
if not os.path.exists(path):
    os.mkdir(path)
else:
    pass
file_writer = open(path_all + "train.csv", 'w')
file_writer.write("file_name,SMILES"+"\n")

@click.command()
@click.option('--group', default=4, help='group number')

def making_data(group):
    print("group number:", group)
    count = 0
    filtered_df = pd.read_csv(data_path +'/filtered_df_group{}.csv'.format(group))
    data_len = len(filtered_df)
    # for idx in tqdm(range(len(filtered_df[filtered_df['group'] == group]))):
    for idx in range(data_len):
        # idx += 3700000 * (group-1)
        # print('idx:',idx, end='\r')
        # smiles = filtered_df[filtered_df['group'] == group]['SMILES'][idx]
        smiles = filtered_df['SMILES'][idx]  # this is the representation string
        if len(smiles) <= 100:
            count += 1
            img_name = str(idx) + ".png"
            #if len(smiles) > 100: continue  # we only choose the samples that are less than 101


            smiles_g = Chem.MolFromSmiles(smiles)
            try:
                # smile_plt is the image so we can directly save it.
                smile_plt = Draw.MolToImage(smiles_g, size = (300,300))

                img_full_name = os.path.join(path, img_name)
                file_writer.write(img_name + "," + smiles + "\n")
                smile_plt.save(img_full_name)  # save the image in png
                # np.save(path  + str(dir_name) + '/' +'{0}_train_{1}_{2}.npy'.format(dir_name, group, idx), arr = convert)
                assert len(smiles) <= 100
                del (smile_plt)
            except ValueError:
                pass

        # checking the completion
        if idx % 10000 == 0 :
            print('group : {0}, index : {1}'.format(group, idx))
    print("Number of length <=100 is {0}".format(count))
    del(filtered_df)
    file_writer.close()


if __name__ == '__main__':
    making_data()

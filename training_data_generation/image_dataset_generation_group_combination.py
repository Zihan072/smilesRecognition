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
from tqdm import tqdm
import click

import warnings
warnings.filterwarnings(action = 'ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# path
path = './new_images_5M/' # Saving new data
if not os.path.exists(path):
    os.mkdir(path)
else:
    pass


img_path = 'new_images_5M/train_img' # Saving new image

data_path = 'train_dataset_3rd_5M'
if not os.path.exists(img_path):
    os.mkdir(img_path)
else:
    pass

# Oragainizing the group by the number of core
# The number of data sameple for one group calculated as
# the number of total data sample / the number of core
# ex) 111307682 / 31 = 3700000
# The number of core can be different by each environment

file_writer = open(path + "/train.csv", 'w')
file_writer.write("file_name,SMILES"+"\n")


@click.command()
@click.option('--group', default=1, help='group number')
group_total = 5
def making_data(group):
    for i in range(group_total):
        print("group number:", group)

        filtered_df = pd.read_csv(data_path +'/filtered_df_group{}.csv'.format(group))
        data_len = len(filtered_df)
        #data_len = 5
        print("data length of this group:", data_len)
        #print("The first line of csv file:", filtered_df[:][:1])
        group += 1
        # for idx in tqdm(range(len(filtered_df[filtered_df['group'] == group]))):
        for idx in tqdm(range(data_len)):
            # idx += 3700000 * (group-1)
            # print('idx:',idx, end='\r')
            # smiles = filtered_df[filtered_df['group'] == group]['SMILES'][idx]
            smiles = filtered_df['SMILES'][idx]  # this is the representation string
            #print(smiles)
            smiles_idx = filtered_df['Unnamed: 0'][idx] #hte index of SMILES
            #print("smiles_idx", smiles_idx)
            img_name = str(smiles_idx) + ".png"
            #print("img_name", img_name)
            #if len(smiles) > 100: continue  # we only choose the samples that are less than 101
            assert len(smiles) <= 100

            smiles_g = Chem.MolFromSmiles(smiles)
            try:
                # smile_plt is the image so we can directly save it.
                smile_plt = Draw.MolToImage(smiles_g, size = (300,300))

                # converting color
                # switching r channel with b channel
                # im = np.array(smile_plt)
                #  r = im[:,:,0]
                # g = im[:,:,1]
                # b = im[:,:,2]
                # convert = np.stack((b,g,r),axis=-1)


                # Making directory by the length of sequnece and saving
                # making filename as "the length of smiels"_train_"index"
                # ex) 0020_train_4
                # dir_name = len(filtered_df['SMILES'][idx])
                # print("dir_name:", dir_name)
                # dir_name = str(dir_name).zfill(4)
                # print("dir_name:", dir_name)
                # os.makedirs(os.path.join(img_path, dir_name), exist_ok=True)
                img_full_name = os.path.join(img_path, img_name)
                #print("img_full_name:", img_full_name)
                file_writer.write(img_name + "," + smiles + "\n") # we must add "," between img_name and smiles, otherwise img_name and smiles will be stored in one column.
                smile_plt.save(img_full_name)  # save the image in png
                # np.save(path  + str(dir_name) + '/' +'{0}_train_{1}_{2}.npy'.format(dir_name, group, idx), arr = convert)


                del (smile_plt)
            except ValueError:
                pass


        # checking the completion
        #if idx % 100000 == 0 :
        #    print('group : {0}, index : {1}'.format(group, idx))

    del(filtered_df)
    file_writer.close()

if __name__ == '__main__':
    making_data()

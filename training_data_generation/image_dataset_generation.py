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
path_all = './new_images_3rd_100/' # Saving new data
if not os.path.exists(path_all):
    os.mkdir(path_all)
else:
    pass


path = path_all + '/train_img' # Saving new image

data_path = 'train_dataset_3rd_100'
if not os.path.exists(path):
    os.mkdir(path)
else:
    pass

# Oragainizing the group by the number of core
# The number of data sameple for one group calculated as
# the number of total data sample / the number of core
# ex) 111307682 / 31 = 3700000
# The number of core can be different by each environment

file_writer = open(path_all + "train.csv", 'w')
file_writer.write("file_name,SMILES"+"\n")


@click.command()
@click.option('--group', default=1, help='group number')

def making_data(group):
    print("group number:", group)

    filtered_df = pd.read_csv(data_path +'/filtered_df_group{}.csv'.format(group))
    data_len = len(filtered_df)
    # for idx in tqdm(range(len(filtered_df[filtered_df['group'] == group]))):
    for idx in range(data_len):
        # idx += 3700000 * (group-1)
        # print('idx:',idx, end='\r')
        # smiles = filtered_df[filtered_df['group'] == group]['SMILES'][idx]
        smiles = filtered_df['SMILES'][idx]  # this is the representation string
        img_name = str(idx) + ".png"
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
            #dir_name = len(filtered_df['SMILES'][idx])
            #dir_name = str(dir_name).zfill(4)
            #os.makedirs(os.path.join(path, dir_name), exist_ok=True)
            img_full_name = os.path.join(path, img_name)
            file_writer.write(img_name + "," + smiles + "\n")
            smile_plt.save(img_full_name)  # save the image in png
            # np.save(path  + str(dir_name) + '/' +'{0}_train_{1}_{2}.npy'.format(dir_name, group, idx), arr = convert)


            del (smile_plt)
        except ValueError:
            pass

        # checking the completion
        if idx % 1000 == 0 :
            print('group : {0}, index : {1}'.format(group, idx))

    del(filtered_df)
    file_writer.close()

if __name__ == '__main__':
    making_data()

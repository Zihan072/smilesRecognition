'''
Origin dataset for Image2InChI is from Bristol-Myers Squibb
which is supported by Kaggle. https://www.kaggle.com/c/bms-molecular-translation/data
In this competition, there are images of chemicals, with the objective of
predicting the corresponding International Chemical Identifier (InChI) text string of the image.
The images provided (both in the training data as well as the test data) may be rotated to different angles,
be at various resolutions, and have different noise levels.
The sizes of images from original dataset are not fixed.
So here I will draw new molecules with 300*300 with RDKit to solve the problems of images of noise.

Author: Zihan Chen
Datum: 19.03.2022
'''
import os.path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from tqdm.auto import tqdm # solve the problem of each iteration of progressbar starts a new line



img_path_origin = './origin_img' # save images from original dataset
img_path_clear = './train_img' #save new generated clear images
img_path_noise = './noise_img'#save new generated images with adding some noises.

if not os.path.exists(img_path_origin):
    os.mkdir(img_path_origin)
else:
    pass

if not os.path.exists(img_path_clear):
    os.mkdir(img_path_clear)
else:
    pass

if not os.path.exists(img_path_noise):
    os.mkdir(img_path_noise)
else:
    pass

path = '/cvhci/temp/zihanchen/data/bms-molecular-translation_InChIs/'
df = pd.read_csv(path + "train_labels.csv")
#df = pd.read_csv("/Users/chenzihan/KIT/MasterThesis/data/bms-molecular-translation_InChIs/train_labels.csv")

#store labels
label_file = open('train.csv', 'w')
label_file.write('file_name,InChI'+'\n')

#print(df.head(5))
#
'''
Draw molecules
'''
# read original image from dataset
img_num = 0
for _, row in df.head(3).iterrows(): #Iterate over DataFrame rows as (index, Series) pairs.
    img_id = row['image_id']
    print('img_id:', img_id)

    #img: original image
    img = cv2.imread(path + "train/{}/{}/{}/{}.png".format(img_id[0], img_id[1], img_id[2], img_id), cv2.IMREAD_GRAYSCALE)
    #Add check if the img exists? if not exists, skip, avoid error.
    #TODO

    #cv2.IMREAD_COLOR: It specifies to load a color image. Any transparency of image will be neglected. It is the default flag. Alternatively, we can pass integer value 1 for this flag.
    #cv2.IMREAD_GRAYSCALE: It specifies to load an image in grayscale mode. Alternatively, we can pass integer value 0 for this flag.
    #cv2.IMREAD_UNCHANGED: It specifies to load an image as such including alpha channel. Alternatively, we can pass integer value -1 for this flag.
    # Could we use grayscale to make model focus on features of structure during training?

    #rename images id
    img_name = str(img_num) + '.png'
    img_num += 1
    img_full_name_origin = os.path.join(img_path_origin, img_name)
    img_full_name_clear = os.path.join(img_path_clear, img_name)
    img_full_name_noise = os.path.join(img_path_noise, img_name)

    # print("img_path+img_name:", img_full_name_origin)
    # print(img_full_name_clear)
    # print(img_full_name_noise)

    cv2.imwrite(img_full_name_origin, img)
    #change the size of img into 300*300

    #test the size of output with cv2.imwrite and plt.savefig()
    #TODO

    #draw new clearer molecule with 0 degree rotation
    InChI = row['InChI']
    mol_inchi = rdkit.Chem.inchi.MolFromInchi(row['InChI'])
    print("InChI sequence:", InChI)

    #draw molecule
    d = rdkit.Chem.Draw.rdMolDraw2D.MolDraw2DCairo(512, 512)
    d.drawOptions().useBWAtomPalette()
    d.drawOptions().rotate = 0
    d.drawOptions().bondLineWidth = 1
    d.DrawMolecule(mol_inchi)
    d.FinishDrawing()
    d.WriteDrawingText("0.png")
    img0 = cv2.imread("0.png", cv2.IMREAD_GRAYSCALE)

    cv2.imwrite(img_full_name_clear, img0)





    # show images
    #plt.figure(figsize=(20, 20))

    #Generate the corresponding smiles sequence.
    #TODO


'''
Draw molecules with noise
we can also draw some molecules with different level of noise.
'''
#TODO

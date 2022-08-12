import os
import argparse
import torch
import torchvision.transforms as transforms

import time
from rdkit import Chem
from rdkit.Chem import Draw
import glob,os

from model.Model import MSTS
from src.datasets import SmilesDataset, PNGSmileDataset
from src.config import input_data_dir, base_file_name, test_dir
from utils import logger, make_directory, load_reversed_token_map, smiles_name_print, str2bool, convert_smiles

#Readme
# # smilesRecognition
# This system suports converting chemical image into SMILES and other chemical representations.
#
# ## Install Enviroment
# conda env create --name chem_info_env --file utils/chem_info.yml
#
# ## Prediction
#



def one_input():
    start_time = time.time()

    #smiles_name_print()


    parser = argparse.ArgumentParser()

    parser.add_argument('--work_type', type=str, default='one_input_pred', help="choose work type which test")
    parser.add_argument('--encoder_type', type=str, default='efficientnetB0',
                        help="choose encoder model type 'efficientnetB2', wide_res', 'res', and 'resnext' ")
    parser.add_argument('--seed', type=int, default=1, help="choose seed number")
    parser.add_argument('--tf_encoder', type=int, default=6, help="the number of transformer layers")
    parser.add_argument('--tf_decoder', type=int, default=6, help="the number of transformer decoder layers")
    parser.add_argument('--decode_length', type=int, default=100, help='length of decoded SMILES sequence')
    parser.add_argument('--emb_dim', type=int, default=512, help='dimension of word embeddings')
    parser.add_argument('--attention_dim', type=int, default=512, help='dimension of attention linear layers')
    parser.add_argument('--decoder_dim', type=int, default=512, help='dimension of decoder RNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='drop out rate')
    parser.add_argument('--device', type=str, default='cuda', help='sets device for model and PyTorch tensors')
    parser.add_argument('--gpu_non_block', type=str2bool, default=True, help='GPU non blocking flag')
    parser.add_argument('--fp16', type=str2bool, default=True, help='Use half-precision/mixed precision training')
    parser.add_argument('--cudnn_benchmark', type=str2bool, default=True,
                        help='set to true only if inputs to model are fixed size; otherwise lot of computational overhead')


    parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--checkpointing_cnn', type=int, default=0, help='Checkpoint  the cnn to save memory')
    parser.add_argument('--workers', type=int, default=8, help='for data-loading; right now, only 1 works with h5py')
    parser.add_argument('--encoder_lr', type=float, default=1e-4, help='learning rate for encoder if fine-tuning')
    parser.add_argument('--decoder_lr', type=float, default=4e-4, help='learning rate for decoder')
    parser.add_argument('--grad_clip', type=float, default=5., help='clip gradients at an absolute value of')
    parser.add_argument('--fine_tune_encoder', type=str2bool, default=True, help='fine-tune encoder')

    parser.add_argument('--model_save_path', type=str, default='graph_save', help='model save path')
    parser.add_argument('--model_load_path', type=str, default='./src/model_path', help='model load path')
    parser.add_argument('--model_load_num', type=int, default=11, help='epoch number of saved model')
    parser.add_argument('--test_file_path', type=str, default=test_dir, help='test file path')
    parser.add_argument('--grayscale', type=str2bool, default=True, help='gray scale images ')
    parser.add_argument('--reversed_token_map_dir)', type=str, default=True, help='gray scale images ')


### Reversed_token_file used to map numbers to string.




    config = parser.parse_args()

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    model = MSTS(config)

    # if config.work_type != 'ensemble_test':
    #     model = MSTS(config) #create one instance of the model





    if config.work_type == 'one_input_pred':
        #TODO shows all possible smiles
        #input one sample for application
        #from src.config import reversed_token_map_dir
        #from .utils import convert_smiles
        #ray.init()

        reversed_token_map_dir = '/cvhci/temp/zihanchen/data/lg_PubChem1M_ChEMBL75_RDkitclear/input_data/REVERSED_TOKENMAP_seed_123_max75smiles.json'
        imgs = glob.glob("./utils/input_img/*.png")
        path = './utils/pred_img/'
        print(imgs)

        for img in imgs:
            print('='*100)
            image = img
            img_name = os.path.basename(image)
            new_img_name = 'new_' + img_name
            print(new_img_name)
            reversed_token_map = load_reversed_token_map(reversed_token_map_dir)

            if config.grayscale is not None:
                transform = transforms.Compose([transforms.Compose([normalize]),
                                                transforms.Grayscale(3)])

            else:
                transform = transforms.Compose([normalize])

            model.model_load()

            smiles = model.one_test(image, reversed_token_map, transform)
            print('Predicted SMILES:', smiles)
            convert_smiles(smiles)
            try:
                img_path = os.path.join(path, new_img_name)
                smiles_g = Chem.MolFromSmiles(smiles)
                # smile_plt is the image so we can directly save it.
                smile_plt = Draw.MolToImage(smiles_g, size = (300,300))
                smile_plt.save(img_path)
            except ValueError:
                print("Predicted SMILES" + smiles + " couldn't be redrawn as chemical structure.")



    else:
        print('incorrect work type received.')

    print('process time:', time.time() - start_time)


    #TODO add one_input test for tool
    # add generating other representations
    # not print model
    # add PubChem to evaluate.
    # elif one_sample_test():
    #     # predict fot image
    #     # read image
if __name__ == '__main__':
    one_input()

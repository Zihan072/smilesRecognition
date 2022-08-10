# smilesRecognition
This system suports converting chemical image into SMILES and other chemical representations.

## Install Enviroment

conda env create --name chem_info_env --file utils/chem_info.yml

## Prediction

Put input images in the folder /model/sutils/input_img



Run commad under path model/

python one_input_pred.py | tee log.csv



Then check predicted information in log.csv and redrawn image in folder /model/sutils/pred_img
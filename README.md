# smilesRecognition
This model suports converting chemical image into SMILES and other chemical representations.





## Implement one sample as input for application

### Download model weights

You can download the model weights from this link

[Dropbox]: https://www.dropbox.com/s/yh8pjj03066t5vi/model_path.zip?dl=0

Place the decompressed file under path:

model/src/model_path



### Install Enviroment



```
conda env create --name chem_info_env --file utils/chem_info.yml
```



### Prediction

Put input images under the folder /model/utils/input_img

Run commad under path model/

```
python one_input_pred.py | tee log.csv
```



The predicted images are save in folder "/model/utils/pred_img"
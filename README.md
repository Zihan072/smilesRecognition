# smilesRecognition
This model suports converting chemical image into SMILES and other chemical representations.





## Implement one sample as input for application

### Download model weights

You can download the model weights from this link

[dropbox](https://www.dropbox.com/s/yh8pjj03066t5vi/model_path.zip?dl=0)

Put the decompressed file under path:  *model/src/model_path*

You can download the model weights of multi-models from this link
#TODO
Put the decompressed file under path:  *model/model/multi-model_path/*


### Install Enviroment



```
conda env create --name chem_info_env --file utils/chem_info.yml
```



### Prediction

Put images which you want to predict under the folder */model/utils/input_img*

Run command under path *model/*

```
python one_input_pred.py | tee log.csv
```


### ensemble prediction(multi-model)

Change test path in ./src/config.py.

Run command 

```
python main.py --work_type ensemble_test
```



You can find the predicted images in folder *"/model/utils/pred_img"*
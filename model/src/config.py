from pathlib import Path

### Data provided by LG
# Head directory
# """train"""
# data_dir = Path('/cvhci/temp/zihanchen/data/new_images_1M_group1')
# train_dir = data_dir / 'train_img' # training images path
# test_dir = '/cvhci/temp/zihanchen/data/DACON_SMILES_data/test/'
# train_csv_dir = data_dir / 'train.csv'
# # train_modified contains information of the train/validation split in a pickle file format. I saved in pickle just for efficiency
# train_pickle_dir = data_dir /'train_modified.pkl'
# # Sample submission directory
# sample_submission_dir = data_dir /'sample_submission.csv'
#
# ### Data directory generated by us
# input_data_dir = data_dir / 'input_data' # output path
# base_file_name = 'seed_910_max100smiles'
#
# ### seed for train/val split
# random_seed = 910
#
# ### Reversed_token_file used to map numbers to string.
# reversed_token_map_dir = input_data_dir/ f'REVERSED_TOKENMAP_{base_file_name}.json'



"""test"""
data_dir = Path('/cvhci/temp/zihanchen/data/DACON_SMILES_data/') #dataset which we created inputfile .json
### Data directory generated by us
input_data_dir = data_dir / 'input_data' # output path
base_file_name = 'seed_123_max75smiles'
### seed for train/val split
random_seed = 123
### Reversed_token_file used to map numbers to string.
reversed_token_map_dir = input_data_dir/ f'REVERSED_TOKENMAP_{base_file_name}.json'

###Test fromPubChem
test_dir = '/cvhci/temp/zihanchen/data/testset_isomeric/test_img_20K/'
sample_submission_dir = '/cvhci/temp/zihanchen/data/testset_isomeric/test_20K.csv'
generate_submission_dir = 'test_20K.csv'
sample_submission_labels_dir = '/cvhci/temp/zihanchen/data/testset_isomeric/test_20K_labels.csv'

# ###submission test
# test_dir = '/cvhci/temp/zihanchen/data/DACON_SMILES_data/test/'
# sample_submission_dir = '/cvhci/temp/zihanchen/data/DACON_SMILES_data/sample_submission.csv'
# generate_submission_dir = 'sample_submission.csv'



"""train(DACON)"""
# data_dir = Path('/cvhci/temp/zihanchen/data/DACON_SMILES_data/')
# train_dir = data_dir / 'train' # training images path
# #test_dir = '/cvhci/temp/zihanchen/data/DACON_SMILES_data/test/'
# train_csv_dir = data_dir / 'train.csv'
# # train_modified contains information of the train/validation split in a pickle file format. I saved in pickle just for efficiency
# train_pickle_dir = data_dir /'train_modified.pkl'
# # Sample submission directory
# sample_submission_dir = data_dir /'sample_submission.csv'
#
# ### Data directory generated by us
# input_data_dir = data_dir / 'input_data' # output path
# base_file_name = 'seed_123_max75smiles'
#
# ### seed for train/val split
# random_seed = 123
#
# ### Reversed_token_file used to map numbers to string.
# reversed_token_map_dir = input_data_dir/ f'REVERSED_TOKENMAP_{base_file_name}.json'


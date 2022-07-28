import os

input_file = '/cvhci/temp/zihanchen/data/new_images_5M/train.csv'
input_dir = '/cvhci/temp/zihanchen/data/RDkit_SMILES_gray/clear_img/'

output_dir = "test_RDKit"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    pass

i = 0
for line in open(input_file).readlines():
    i = i+1
    if i == 1: continue
    file_name = line.split(",")[0]
    src_file = os.path.join(input_dir, file_name)
    print(src_file)
    #shutils.copy(src_file, "test_img")
    #os.system("ln -s %s %s" % (src_file, "test_img_50K"))
    os.system("cp %s %s" % (src_file, "test_img_20K"))

    os.system("rm %s" % (src_file))

'''example:
train.csv file store tail 50k but drop last 10k lines, save in train_50K.csv
head -n 1  train.csv > train_50K.csv #store head of csv file

Tail -n 50000  train.csv｜ head -n 40000   >> train_50K.csv 

#remove 3000 last line
tail -n 3000 input.csv >> output.csv
head -n +3000 input.csv > input.csv.truncated   #head -n +number returns everything but the n last line. ps:dosen't work, 
#just use head -n 955820 instead of it 
mv input.csv.truncated input.csv



































'''
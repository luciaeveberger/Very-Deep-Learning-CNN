import numpy as np # linear algebra
import pandas as pd
import shutil
import os
from os import listdir, makedirs
from os.path import join, exists, expanduser
from keras.preprocessing import image
from sklearn.model_selection import train_test_split


data_dir="./"
train_data_dir= './data/train/'
labels = pd.read_csv(join(data_dir, 'labels.csv'))
sample_submission = pd.read_csv(join(data_dir, 'sample_submission.csv'))
num_classes = 120
SEED = 1987
 
selected_breed_list = list(labels.groupby('breed').count().sort_values(by='id', ascending=False).head(num_classes).index)
labels = labels[labels['breed'].isin(selected_breed_list)]
labels['target'] = 1
group = labels.groupby(by='breed', as_index=False).agg({'id': pd.Series.nunique})
group = group.sort_values('id',ascending=False)
 
 
labels_pivot = labels.pivot('id', 'breed', 'target').reset_index().fillna(0)
np.random.seed(seed=SEED)
for breed in group.get_values():
    directory =  train_data_dir+breed[0];
    if not os.path.exists(directory):
        os.makedirs(directory)
         
fileExtension = str("jpg")
files = [i for i in os.listdir(train_data_dir+"dogs")]
files.remove('.DS_Store')
print(files)
for file in files:
     fileget = labels.loc[labels['id'] == file[:-4], 'breed'].values[0]
#      shutil.copy2(train_data_dir+"dogs/"+file, train_data_dir+fileget+'/' +file)

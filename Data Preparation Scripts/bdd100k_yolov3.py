import os
import sys
import pandas as pd
import json

os.chdir(r'./data/bdd100k/bdd100k/images/10k')

print(os.getcwd())

os.chdir(r'train')
train_images_list = os.listdir()
os.chdir(r'..')
os.chdir(r'test')
test_images_list = os.listdir()
os.chdir(r'..')
os.chdir(r'val')
val_images_list = os.listdir()
os.chdir(r'..')

os.chdir(r'../../../..')
print(os.getcwd())

os.chdir(r'./bdd100k_labels_release/bdd100k/labels')
with open('bdd100k_labels_images_train.json', 'r') as file:
    data_train_labels = json.load(file)
with open('bdd100k_labels_images_val.json', 'r') as file:
    data_val_labels = json.load(file)

#df_train_labels = pd.read_json('bdd100k_labels_images_train.json')
df_train_labels_norm = pd.json_normalize(data_train_labels)

os.chdir('../../..')
try:
    os.mkdir('labels')
except:
    pass
os.chdir('labels')

try:
    os.mkdir('train')
except:
    pass
os.chdir('train')
#df_train_labels_final = pd.DataFrame(columns=['name','id','category','box2d'])
df_train_labels_final = pd.DataFrame(columns=['name','id','category','x_center','y_center','width','height'])
for idx, row in df_train_labels_norm.iterrows():
    df_labels = pd.DataFrame(row['labels'])
    df_labels = df_labels[~df_labels['box2d'].isnull()]
    #df_labels = df_labels[['id','category','box2d']]
    df_labels['name'] = row['name']
    df_labels['x_center'] = df_labels['box2d'].apply(lambda x: (x["x1"]+x["x2"])/2)
    df_labels['y_center'] = df_labels['box2d'].apply(lambda x: (x["y1"]+x["y2"])/2)
    df_labels['width'] = df_labels['box2d'].apply(lambda x: x["x2"]-x["x1"])
    df_labels['height'] = df_labels['box2d'].apply(lambda x: x["y2"]-x["y1"])
    df_labels = df_labels[['name','id','category','x_center','y_center','width','height']]
    df_train_labels_final = pd.concat([df_train_labels_final, df_labels], axis=0)
    #df_labels[['id',]].to_csv('bdd100k_train_'+row['name'][:-4]+'.txt', sep='\t', header=False, index=False)
    if idx%100 == 0:
        print(idx)

classes = df_train_labels_final['category'].unique()
df_classes = pd.DataFrame(data=[[x] for x in classes], columns=['category']).reset_index(drop=False).rename({'index':'class'}, axis=1)
df_train_labels_class = pd.merge(left=df_train_labels_final, right=df_classes, on='category', how='left')

df_train_labels_class_bkup = df_train_labels_class.copy()

df_train_labels_class['x_center'] /= 1280
df_train_labels_class['width'] /= 1280
df_train_labels_class['y_center'] /= 720
df_train_labels_class['height'] /= 720

df_train_labels_class.x_center = df_train_labels_class.x_center.round(6)
df_train_labels_class.width = df_train_labels_class.width.round(6)
df_train_labels_class.y_center = df_train_labels_class.y_center.round(6)
df_train_labels_class.height = df_train_labels_class.height.round(6)

i = 0
for name, df_group in df_train_labels_class.groupby('name'):
    df_group = df_group[['class','x_center','y_center','width','height']]
    df_group.to_csv(name[:-4]+'.txt', sep=' ', index=False, header=False)
    i+=1
    if i%100 == 0:
        print(i)

#******************************** TRAIN COMPLETE *****************************************

df_val_labels_norm = pd.json_normalize(data_val_labels)

os.chdir('..')
try:
    os.mkdir('val')
except:
    pass
os.chdir('val')
df_val_labels_final = pd.DataFrame(columns=['name','id','category','x_center','y_center','width','height'])
for idx, row in df_val_labels_norm.iterrows():
    df_labels = pd.DataFrame(row['labels'])
    df_labels = df_labels[~df_labels['box2d'].isnull()]
    #df_labels = df_labels[['id','category','box2d']]
    df_labels['name'] = row['name']
    df_labels['x_center'] = df_labels['box2d'].apply(lambda x: (x["x1"]+x["x2"])/2)
    df_labels['y_center'] = df_labels['box2d'].apply(lambda x: (x["y1"]+x["y2"])/2)
    df_labels['width'] = df_labels['box2d'].apply(lambda x: x["x2"]-x["x1"])
    df_labels['height'] = df_labels['box2d'].apply(lambda x: x["y2"]-x["y1"])
    df_labels = df_labels[['name','id','category','x_center','y_center','width','height']]
    df_val_labels_final = pd.concat([df_val_labels_final, df_labels], axis=0)
    #df_labels[['id',]].to_csv('bdd100k_train_'+row['name'][:-4]+'.txt', sep='\t', header=False, index=False)
    if idx%100 == 0:
        print(idx)

classes = df_val_labels_final['category'].unique()
#df_classes = pd.DataFrame(data=[[x] for x in classes], columns=['category']).reset_index(drop=False).rename({'index':'class'}, axis=1)
df_val_labels_class = pd.merge(left=df_val_labels_final, right=df_classes, on='category', how='left')

df_val_labels_class_bkup = df_val_labels_class.copy()

df_val_labels_class['x_center'] /= 1280
df_val_labels_class['width'] /= 1280
df_val_labels_class['y_center'] /= 720
df_val_labels_class['height'] /= 720

df_val_labels_class.x_center = df_val_labels_class.x_center.round(6)
df_val_labels_class.width = df_val_labels_class.width.round(6)
df_val_labels_class.y_center = df_val_labels_class.y_center.round(6)
df_val_labels_class.height = df_val_labels_class.height.round(6)

i = 0
for name, df_group in df_val_labels_class.groupby('name'):
    df_group = df_group[['class','x_center','y_center','width','height']]
    df_group.to_csv(name[:-4]+'.txt', sep=' ', index=False, header=False)
    i+=1
    if i%100 == 0:
        print(i)

#******************************** VALIDATION COMPLETE *****************************************

os.chdir('../../..')
df_classes[['category']].to_csv('bdd100k.names', index=False, header=False)

#********************************* CLASSES COMPLETE *******************************************

df_train_labels_norm.to_csv('df_train_labels_norm.csv', index=False)
df_val_labels_norm.to_csv('df_val_labels_norm.csv', index=False)
df_train_labels_class.to_csv('df_train_labels_class.csv', index=False)
df_val_labels_class.to_csv('df_val_labels_class.csv', index=False)





import os
import sys
import pandas as pd
import json

os.getcwd()

with open('./yolov7/coco/annotations/instances_val2017.json', 'r') as file:
    data_coco_labels = json.load(file)

level_0 = data_coco_labels.copy()

df_coco_labels_norm = pd.json_normalize(level_0)

print(df_coco_labels_norm)

level_1 = df_coco_labels_norm.loc[0,'annotations']

df_coco_labels_norm_2 = pd.json_normalize(level_1)

print(df_coco_labels_norm_2)

level_2 = df_coco_labels_norm_2.loc[0,'bbox']

df_coco_labels_norm_3 = pd.json_normalize(level_2)

print(df_coco_labels_norm_3) # empty, so only 2 levels of depth

#level_3 = df_coco_labels_norm_3.iloc[0,1]

data_mod_labels = data_coco_labels.copy()

data_mod_labels['licenses'] = []
data_mod_labels['info']['description'] = ''
data_mod_labels['info']['url'] = ''
data_mod_labels['info']['version'] = ''
data_mod_labels['info']['year'] = ''
data_mod_labels['info']['contributor'] = ''
data_mod_labels['info']['date_created'] = ''
data_mod_labels['images'] = [] # Need to add 'file_name', 'height', 'width' and 'id' for each image (dict)
data_mod_labels['annotations'] = [] # Need to add level 1: 'image_id', 'bbox', 'category_id' with data 
# and 'segmentation', 'area', 'iscrowd' without. Also level 2 in 'bbox': 4 coordinates of diagonals x1, y1, x2, y2 for each category_id (dict)
data_mod_labels['categories'] = [] # Need to add 'id', 'name' with data and 'supercategory' without data for each category (dict)

print(data_mod_labels)

os.chdir(r'data/bdd100k_labels_release/bdd100k/labels')
with open('bdd100k_labels_images_train.json', 'r') as file:
    data_train_labels = json.load(file)
with open('bdd100k_labels_images_val.json', 'r') as file:
    data_val_labels = json.load(file)

os.chdir('../../../..')

df_train_labels_norm = pd.json_normalize(data_train_labels)

df_train_labels_final = pd.DataFrame(columns=['name','id','category','x1','y1','x2','y2'])
for idx, row in df_train_labels_norm.iterrows():
    df_labels = pd.DataFrame(row['labels'])
    df_labels = df_labels[~df_labels['box2d'].isnull()]
    df_labels['name'] = row['name']
    df_labels['id'] = idx
    df_labels['x1'] = df_labels['box2d'].apply(lambda x: x['x1'])
    df_labels['y1'] = df_labels['box2d'].apply(lambda x: x['y1'])
    df_labels['x2'] = df_labels['box2d'].apply(lambda x: x['x2'] - x['x1']) # width
    df_labels['y2'] = df_labels['box2d'].apply(lambda x: x['y2'] - x['y1']) # height
    df_labels = df_labels[['name','id','category','x1','y1','x2','y2']]
    df_train_labels_final = pd.concat([df_train_labels_final, df_labels], axis=0)
    if idx%100 == 0:
        print(idx)

classes = df_train_labels_final['category'].unique()
df_classes = pd.DataFrame(data=[[x] for x in classes], columns=['category']).reset_index(drop=False).rename({'index':'category_id'}, axis=1)
df_train_labels_class = pd.merge(left=df_train_labels_final, right=df_classes, on='category', how='left')

df_train_labels_class_bkup = df_train_labels_class.copy()

df_train_labels_class.x1 = df_train_labels_class.x1.round(2)
df_train_labels_class.y1 = df_train_labels_class.y1.round(2)
df_train_labels_class.x2 = df_train_labels_class.x2.round(2)
df_train_labels_class.y2 = df_train_labels_class.y2.round(2)

# Populating 'images'
data_bdd_train_labels = data_mod_labels.copy()
data_bdd_train_images = df_train_labels_class[['name','id']].drop_duplicates().rename({'name':'file_name'},axis=1)
#data_bdd_train_images['height'] = 720
#data_bdd_train_images['width'] = 1280
#data_bdd_train_labels['images'] = data_bdd_train_images.to_dict('records')
data_bdd_train_labels['images'] = [{'file_name':data_bdd_train_images.iloc[i,0], #.item(), 
                                    'id':data_bdd_train_images.iloc[i,1], #.item(), 
                                    'height':720,
                                    'width':1280} for i in range(len(data_bdd_train_images))]

# Populating 'annotations'
data_bdd_train_annotations = df_train_labels_class[['x1','y1','x2','y2','category_id','id']].reset_index(drop=False) # Removed 'id' to calculate below, added it back
#data_bdd_train_annotations.rename({'index':'id'}, axis=1, inplace=True)
data_bdd_train_labels['annotations'] = [{'image_id': data_bdd_train_annotations.loc[i,'id'], #.item(), # This is in numpy.int64
                                         'bbox': [data_bdd_train_annotations.loc[i,'x1'].item(), # All 4 are in numpy.int64
                                                  data_bdd_train_annotations.loc[i,'y1'].item(), 
                                                  data_bdd_train_annotations.loc[i,'x2'].item(), 
                                                  data_bdd_train_annotations.loc[i,'y2'].item()],
                                         'category_id':data_bdd_train_annotations.loc[i,'category_id'].item(),
                                         'id':data_bdd_train_annotations.loc[i,'index'].item(), # Annotation ID
                                         'area':data_bdd_train_annotations.loc[i,'x2'].item()*data_bdd_train_annotations.loc[i,'y2'].item(), # Rectangle area
                                         'segmentation':[], 'iscrowd':0} for i in range(len(data_bdd_train_annotations))] # Convert 'iscrowd' = 0

# Populating 'categories'
df_train_classes = df_classes.copy()
df_train_classes['supercategory'] = ''
df_train_classes.rename({'category_id':'id', 'category':'name'}, axis=1, inplace=True)
#data_bdd_train_labels['categories'] = df_train_classes.to_dict('records')
data_bdd_train_labels['categories'] = [{'id': df_train_classes.loc[i,'id'].item(), # This is in np.int64
                                        'name': df_train_classes.loc[i,'name'], #.item(), 
                                        'supercategory':''} for i in range(len(df_train_classes))]

str_bdd_train_labels = json.dumps(data_bdd_train_labels)

file_name = "instances_bdd_train.json"
try:
    with open(file_name, "w") as file:
        file.write(str_bdd_train_labels)
    print(f"String successfully written to '{file_name}'")
except IOError as e:
    print(f"Error writing to file: {e}")

# *************************************** TRAIN COMPLETE *********************************************


df_val_labels_norm = pd.json_normalize(data_val_labels)

df_val_labels_final = pd.DataFrame(columns=['name','id','category','x1','y1','x2','y2'])
for idx, row in df_val_labels_norm.iterrows():
    df_labels = pd.DataFrame(row['labels'])
    df_labels = df_labels[~df_labels['box2d'].isnull()]
    df_labels['name'] = row['name']
    df_labels['id'] = idx
    df_labels['x1'] = df_labels['box2d'].apply(lambda x: x['x1'])
    df_labels['y1'] = df_labels['box2d'].apply(lambda x: x['y1'])
    df_labels['x2'] = df_labels['box2d'].apply(lambda x: x['x2'] - x['x1']) # width
    df_labels['y2'] = df_labels['box2d'].apply(lambda x: x['y2'] - x['y1']) # height
    df_labels = df_labels[['name','id','category','x1','y1','x2','y2']]
    df_val_labels_final = pd.concat([df_val_labels_final, df_labels], axis=0)
    if idx%100 == 0:
        print(idx)

#classes = df_train_labels_final['category'].unique()
#df_classes = pd.DataFrame(data=[[x] for x in classes], columns=['category']).reset_index(drop=False).rename({'index':'category_id'}, axis=1)
df_val_labels_class = pd.merge(left=df_val_labels_final, right=df_classes, on='category', how='left')

df_val_labels_class_bkup = df_val_labels_class.copy()

df_val_labels_class.x1 = df_val_labels_class.x1.round(2)
df_val_labels_class.y1 = df_val_labels_class.y1.round(2)
df_val_labels_class.x2 = df_val_labels_class.x2.round(2)
df_val_labels_class.y2 = df_val_labels_class.y2.round(2)

# Populating 'images'
data_bdd_val_labels = data_mod_labels.copy()
data_bdd_val_images = df_val_labels_class[['name','id']].drop_duplicates().rename({'name':'file_name'},axis=1)
data_bdd_val_labels['images'] = [{'file_name':data_bdd_val_images.iloc[i,0], #.item(), 
                                  'id':data_bdd_val_images.iloc[i,1], #.item(), 
                                  'height':720,
                                  'width':1280} for i in range(len(data_bdd_val_images))]

# Populating 'annotations'
data_bdd_val_annotations = df_val_labels_class[['x1','y1','x2','y2','category_id','id']].reset_index(drop=False) # Removed 'id' to calculate below, added it back
#data_bdd_val_annotations.rename({'index':'id'}, axis=1, inplace=True)
data_bdd_val_labels['annotations'] = [{'image_id': data_bdd_val_annotations.loc[i,'id'], #.item(),  # This is in numpy.int64
                                       'bbox': [data_bdd_val_annotations.loc[i,'x1'].item(), # All 4 are in numpy.int64
                                                data_bdd_val_annotations.loc[i,'y1'].item(), 
                                                data_bdd_val_annotations.loc[i,'x2'].item(), 
                                                data_bdd_val_annotations.loc[i,'y2'].item()],
                                        'category_id':data_bdd_val_annotations.loc[i,'category_id'].item(), 
                                        'id':data_bdd_val_annotations.loc[i,'index'].item(), # Annotation ID
                                        'area':data_bdd_val_annotations.loc[i,'x2'].item()*data_bdd_val_annotations.loc[i,'y2'].item(), # Rectangle area
                                        'segmentation':[], 'iscrowd':0} for i in range(len(data_bdd_val_annotations))] # Convert 'iscrowd' = 0

# Populating 'categories'
df_val_classes = df_classes.copy()
df_val_classes['supercategory'] = ''
df_val_classes.rename({'category_id':'id', 'category':'name'}, axis=1, inplace=True)
data_bdd_val_labels['categories'] = [{'id': df_val_classes.loc[i,'id'].item(), # This is in np.int64
                                      'name': df_val_classes.loc[i,'name'], #.item(), 
                                      'supercategory':''} for i in range(len(df_val_classes))]

str_bdd_val_labels = json.dumps(data_bdd_val_labels)

file_name = "instances_bdd_val.json"
try:
    with open(file_name, "w") as file:
        file.write(str_bdd_val_labels)
    print(f"String successfully written to '{file_name}'")
except IOError as e:
    print(f"Error writing to file: {e}")

# ************************************* VALIDATION COMPLETE *********************************************
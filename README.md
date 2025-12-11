# ece5554-computer-vision-cs5814-intro-to-deep-learning-project

##Detectron 2 Notebooks:
Included are the .ipynb files for the Faster R-CNN and RetinaNet with complete step-by-step cell blocks for importing datasets, registering them and running the models.

##Custom PyTorch:
Please use the code as is from <https://github.com/eriklindernoren/PyTorch-YOLOv3>. Included in this repo are scripts for generating the annotations compatible with this and Detectron 2 in the 'Data Engineering' subfolder.

##Reference links:
Project site: [https://sites.google.com/view/vt-ece-5554-fall-2025-group-22/](https://sites.google.com/view/vt-ece-5554-fall-2025-group-22/)
Dataset site: [https://bair.berkeley.edu/blog/2018/05/30/bdd/](https://bair.berkeley.edu/blog/2018/05/30/bdd/)
              [https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k)
Report:       Present in the parent directory

##Download links (will be active until Dec 31st, 2025):
BDD100K:      [https://dlcv-project.s3.us-east-1.amazonaws.com/bdd100k.zip](https://dlcv-project.s3.us-east-1.amazonaws.com/bdd100k.zip)
Detectron 2's [https://dlcv-project.s3.us-east-1.amazonaws.com/instances_bdd_train.json](https://dlcv-project.s3.us-east-1.amazonaws.com/instances_bdd_train.json)
Annotations:  [https://dlcv-project.s3.us-east-1.amazonaws.com/instances_bdd_val.json](https://dlcv-project.s3.us-east-1.amazonaws.com/instances_bdd_val.json)

##System Requirements: 
IPython notebooks need to be run on Google Colab with atleast T4 GPUs selected. The Custom PyTorch implementation is recommended to be run on Linux Debian environment (used for this project).

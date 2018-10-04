
# coding: utf-8

# # Mask R-CNN - Train on Shapes Dataset
# 
# 
# This notebook shows how to train Mask R-CNN on your own dataset. To keep things simple we use a synthetic dataset of shapes (squares, triangles, and circles) which enables fast training. You'd still need a GPU, though, because the network backbone is a Resnet101, which would be too slow to train on a CPU. On a GPU, you can start to get okay-ish results in a few minutes, and good results in less than an hour.
# 
# The code of the *Shapes* dataset is included below. It generates images on the fly, so it doesn't require downloading any data. And it can generate images of any size, so we pick a small image size to train faster. 

# In[17]:


import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
from config import Config
import utils
import model as modellib
import visualize
from model import log
#import coco
import pandas as pd
import json
from sklearn.cross_validation import train_test_split
from PIL import Image
import skimage.io

#get_ipython().run_line_magic('matplotlib', 'inline')

# Root directory of the project
ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.abspath(os.path.join(ROOT_DIR, os.pardir)) #this is infact the parent of the current dir

timestr = time.strftime("%Y%m%d-%H%M%S")
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
#print(ROOT_DIR, MODEL_DIR, COCO_MODEL_PATH)
ARF_MEAN_2013_UT33_DIR = "/arf/mean.js/public/2013/UT33-13D/Daily Photos" #includes all the photos for 2013
imgs_loc = sys.argv[1] # where the training images are located
tags_info = sys.argv[2] # where the csv file that containes tags are located
layers = sys.argv[3] # what layers you want to train. options are: all, heads, 3+, 4+ and 5+
ep = sys.argv[4] # the number of epochs
lr = sys.argv[5]

SIZE = sys.argv[6] #128

parametrs = imgs_loc + '\n' + tags_info + '\n' + layers + '\n'+ ep + '\n' + lr + '\n' + SIZE
print(parametrs)
'''
file = open(ROOT_DIR + '/scavenging/parameters_' + timestr, 'w')
file.write(parametrs)
file.close()
'''
ep = int(ep)
SIZE = int(SIZE)

## All other photos for other years are in the public directory
## A file named tags1.csv in /home/wyan3/origin has all the tags for all of the photos for all years


# ## Configurations


class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "corpse"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8
    LEARNING_RATE = float(lr) #0.02

    # Number of classes (including background)
    ##used this to get the number of unique classes
    ##df = pd.read_csv('/home/wyan3/original/tags1.csv'),  df['tag'].unique().shape[0], 
    ## The answer was 298
    NUM_CLASSES = 2 #4

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = SIZE #448 #832
    IMAGE_MAX_DIM = SIZE #704 #1216
    
    TRAIN_ROIS_PER_IMAGE = 32

    # Use smaller anchors because our image and objects are small
    ##RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    ## I comment this out because it seems the one defines in the config class is more generic

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    ##TRAIN_ROIS_PER_IMAGE = 32  ## RoiAlign - Realigning RoIPool to be More Accurate

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1#100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1#80
    ## comments with 2 # are Sara's
    
config = ShapesConfig()
#config.display()


# ## Notebook Preferences

# In[19]:


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.all_sub_imgsplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


# ## Dataset
# 
# Create a synthetic dataset
# 
# Extend the Dataset class and add a method to load the shapes dataset, `load_shapes()`, and override the following methods:
# 
# * load_image()
# * load_mask()
# * image_reference()

# In[20]:
#this function needs to be changed. Meaning it should divide based on the name of the images not the csv file.
#with the current implementation one part of an image might be in train and another part of it in val
def split_data(csv_file):
    df = pd.read_csv(csv_file)
    train,test = train_test_split(df, test_size = 0.3)
    train = train.reset_index(drop = True)
    test = test.reset_index(drop = True)
    #train.to_csv(ROOT_DIR + '/train_data_' + timestr + '.csv')
    #test.to_csv(ROOT_DIR + '/val_data_' + timestr + '.csv')
    train.to_csv(ROOT_DIR + '/train_data_mum_unnorm1000.csv')
    test.to_csv(ROOT_DIR + '/val_data_mum_unnorm_1000.csv')
    return train, test

class ForensicDataset(utils.Dataset):
    def __init__(self, data, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        #self.class_info = [{"source": "", "id": 0, "name": "BG"}] ##Not sure if I need to keep this
        self.class_info = []
        self.add_class("forensics", 0, 'BG')
        self.add_class("forensics", 1, 'mummification')
        #self.add_class("forensics", 2, 'mold')
        #self.add_class("forensics", 3, 'purge')
        self.source_class_ids = {}
        self.data = data

    def load_sub_images(self):
        df = self.data 
        #num_unique_tags = df.tag.unique().shape[0]

        imgs = [] # a list of images in which we have the taged images and their locations
        image_dir = imgs_loc
        for i in range(df['image1'].unique().shape[0]):
            all_sub_imgs = df.loc[df['image1'] == df['image1'].unique()[i]]
            all_sub_imgs = all_sub_imgs.reset_index(drop = True)
            images_dir = image_dir + df['image1'].unique()[i]
            if os.path.isfile(images_dir):
                im = skimage.io.imread(images_dir)
                im, window, scale, padding = utils.resize_image(im,
                        min_dim=config.IMAGE_MIN_DIM,
                        max_dim=config.IMAGE_MAX_DIM,
                        padding=config.IMAGE_PADDING)
                #skimage.io.imsave('t.png', im)
                '''
                it returns:
                image: the resized image
                window: (y1, x1, y2, x2). If max_dim is provided, padding might
                be inserted in the returned image. If so, this window is the
                coordinates of the image part of the full image (excluding
                the padding). The x2, y2 pixels are not included.
                scale: The scale factor used to resize the image
                padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
                '''
                hei, wi = im.shape[:2] #im.se
                window_w = window[3] - window[1]
                window_h = window[2] - window[0]
                for j in range(all_sub_imgs.shape[0]):
                    sub_info = all_sub_imgs['location'][j]
                    sub_info = json.loads(sub_info)[0]
                    geometry = sub_info['geometry']
                    x = int(float(geometry['x']) * wi)
                    y = int(float(geometry['y']) * hei)
                    w = int(float(geometry['width']) * wi)
                    h = int(float(geometry['height']) * hei)

                    '''
                    x = int(float(geometry['x']) * window_w + padding[1][0]) #padding[1][0] is what's padded to the left
                    y = int(float(geometry['y']) * window_h + padding[0][0]) #padding[0][0] is what's added to the top
                    w = int(float(geometry['width']) * window_w)
                    h = int(float(geometry['height']) * window_h)
                    '''

                    sub_info = (x, y, w, h)
                    imgs.append((all_sub_imgs['tag'][j], sub_info))

                self.add_image("forensics", image_id = i, path = images_dir,width = wi, height= hei, images = imgs)
            image_dir = imgs_loc
            imgs = []
            #print(self.image_info)
            #break


    #this method is going to load/read the image(that includes the sub_imges)
    #Haven't done anything with dtype. I might need to change it to uint8
    def load_image(self, image_id):
        #print(self.image_info[image_id]["path"])
        path = self.image_info[image_id]["path"]
        tag = self.image_info[image_id]["images"] 
        #print("id : ", image_id, " tag: ", tag, " path: ", path)
        #image = cv2.imread(path,1) #1 loads a color image
        image = cv2.imread(path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
        return image
        '''
        if tag[0][0] == 'fly':
            img_array = np.zeros((128, 128, 3))
        elif tag[0][0] == 'larvae':
            img_array = np.ones((128, 128, 3))
        return img_array
        '''
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        #print("info: ", info)
        shapes = info['images']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, dims) in enumerate(info['images']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)

        mask_temp = mask.copy()
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            if (mask[:, :, i] * occlusion).any():
                mask[:, :, i] = mask[:, :, i] * occlusion
                occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        if mask[:, :, 0].sum() == 0:
            import bpython
            bpython.embed(locals())
        if mask.sum() == 0:
            print("The mask was all zeros")
        #print("output of the load_mask:", mask, class_ids.astype(np.int32))
        return mask, class_ids.astype(np.int32)

    def draw_shape(self, image, shape, dims, color): 
        """Draws a shape from the given specs."""
        x, y, w, h = dims
        #print("mask dimentions:", int(x), int(y), int(w), int(h))
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), color, -1)
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        #print(info)
        if info["source"] == "forensics":
            return info["images"]
        else:
            super(self.__class__).image_reference(self, image_id)

########################################################
###################### MAIN ############################
########################################################
#train, test = split_data(tags_info)

train = pd.read_csv(ROOT_DIR+ '/mumm.train.data.csv')
test = pd.read_csv(ROOT_DIR + '/mumm.val.data.csv')
# Training dataset


'''
class TrainConfig(ShapesConfig):
    STEPS_PER_EPOCH = train.shape[0] // (GPU_COUNT * IMAGES_PER_GPU)
    VALIDATION_STEPS = test.shape[0] // (GPU_COUNT * IMAGES_PER_GPU)

config = TrainConfig()
'''
config.STEPS_PER_EPOCH = train.shape[0] // (config.GPU_COUNT * config.IMAGES_PER_GPU)
config.VALIDATION_STEPS = test.shape[0] // (config.GPU_COUNT * config.IMAGES_PER_GPU)

dataset_train = ForensicDataset(train)
dataset_train.load_sub_images()
dataset_train.prepare()
#image_ids = np.random.choice(dataset_train.image_ids, 4)
# Validation dataset
dataset_val = ForensicDataset(test)
dataset_val.load_sub_images()
dataset_val.prepare()


# Load and display random samples
#image_ids = np.random.choice(dataset_train.image_ids, 4)
'''
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)
#    print(dataset_train.image_reference(image_id))
    dataset_train.image_reference(image_id)
'''
# ## Ceate Model

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "last"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# ## Training
# 
# Train in two stages:
# 1. Only the heads. Here we're freezing all the backbone layers and training only the randomly initialized layers (i.e. the ones that we didn't use pre-trained weights from MS COCO). To train only the head layers, pass `layers='heads'` to the `train()` function.
# 
# 2. Fine-tune all layers. For this simple example it's not necessary, but we're including it to show the process. Simply pass `layers="all` to train all layers.


# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
model.train(dataset_train, dataset_val, learning_rate = config.LEARNING_RATE/10 , 
        epochs = ep, layers = layers)

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.

'''
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=20, 
            layers="all")
'''

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
# model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
# model.keras_model.save_weights(model_path)


# ## Detection

# In[ ]:


class InferenceConfig(ShapesConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)


# ## Evaluation

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
#image_ids = np.random.choice(dataset_val.image_ids, 10)
image_ids = dataset_val.image_ids
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =        utils.compute_ap(gt_bbox, gt_class_id,
                         r["rois"], r["class_ids"], r["scores"])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))


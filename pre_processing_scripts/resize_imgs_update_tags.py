#!/usr/bin/env python3
#how I ran it: python3 resize_imgs_update_tags.py total_replaced_tags.csv all_images/all_imgs/
import numpy as np
import pandas as pd
import glob
import sys
import csv
import cv2
import os
import skimage.io
import scipy.misc
import json

                    
def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding


def load_data_update_tags(data, image_loc, min_dim, max_dim):
    df = data
    df2 = pd.DataFrame(columns=df.columns)
    for i in range(df['image1'].unique().shape[0]):
        all_sub_imgs = df.loc[df['image1'] == df['image1'].unique()[i]]
        all_sub_imgs = all_sub_imgs.reset_index(drop = True)
        image_filename = image_loc + df['image1'].unique()[i]

        if os.path.isfile(image_filename):
            im = skimage.io.imread(image_filename)
            im, window, scale, padding = resize_image(im,
                    min_dim=min_dim,
                    max_dim=max_dim,
                    padding=True)
            
            #skimage.io.imsave(image_filename.replace("all_imgs","resized512_512_all_imgs"), im)

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
                x = int(float(geometry['x']) * window_w + padding[1][0]) #padding[1][0] is what's padded to the left
                y = int(float(geometry['y']) * window_h + padding[0][0]) #padding[0][0] is what's added to the top
                w = int(float(geometry['width']) * window_w)
                h = int(float(geometry['height']) * window_h)

                sub_info['geometry']['x'] = x / min_dim
                sub_info['geometry']['y'] = y / min_dim
                sub_info['geometry']['width'] = w / min_dim
                sub_info['geometry']['height'] = h / min_dim
                all_sub_imgs['location'][j] = json.dumps(sub_info)
                df2 = df2.append(all_sub_imgs[j:j+1],ignore_index=True)
                
        else:
            sys.stderr.write("Cannot read " + image_filename + "\n")

    df2.to_csv("normalized512by512_tags.csv", index = False)
#for name in glob.glob(sys.argv[1]+"*.JPG"):

if __name__=="__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 resize_imgs_update_tags.py total_replaced_tags.csv all_images/all_imgs/")
        exit()

    data = pd.read_csv(sys.argv[1])
    imgs_loc = sys.argv[2]
    load_data_update_tags(data, imgs_loc, 512, 512)   

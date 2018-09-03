#This script resizes the short side of all imges in <where_to_read> to new_size

import csv
import os.path
import PIL
from PIL import Image
import glob
import sys

if __name__=='__main__':

    where_to_read = sys.argv[1]
    #'/home/mousavi/Mask_RCNN_Sara/few_imgs_and_tags/all_images/all_imgs/'
    resized_path = sys.argv[2]
    #'/home/mousavi/Mask_RCNN_Sara/few_imgs_and_tags/all_images/all_imgs_resized_256by/'
    new_size = int(sys.argv[3])#256
    new_width = 0
    new_height = 0
    count = 1
    if (os.path.isdir(where_to_read) == False):
        print("The source file does not exist")
        exit()
    if (os.path.isdir(resized_path) == False):
        print("The destination file does not exist")
        exit()

    #Reading the image names from a directory
    for image_name in glob.glob(where_to_read + "/*.JPG"):
        readimg = Image.open(image_name)
        print(count)
        count += 1
        name = image_name[image_name.find(where_to_read + "/")+len(where_to_read + "/"):]
        name = resized_path + "/" + name
        width, height = readimg.size
        if  width <= height:
            width_percent = new_size / float(width)
            new_height = int(float(height) * float(width_percent))
            new_width = new_size
            readimg = readimg.resize((new_width, new_height), PIL.Image.ANTIALIAS)
        elif height <= width:
            height_percent = new_size /float(height)
            new_width = int(float(width) * float(height_percent))
            new_height = new_size
            readimg = readimg.resize((new_width, new_height), PIL.Image.ANTIALIAS)

        readimg.save(name)


'''
#Reading the image names from a csv file
    csv_file = '/home/mousavi/Mask_RCNN_Sara/few_imgs_and_tags/high_frec_tag_data/high_frec_single_word_tag.csv'
    f = open(csv_file)
    data = csv.reader(f)
 
    cropped_imgs_metedata = []
    cropped_imgs_metedata.append(['row_number', 'X_id', 'user', 'location', 'image', 'tag', 'created', 'X__v', 'image1'])
    counter = 0
    im_name = 0
    for row in data:
        counter += 1
        if counter <  2:
            continue
        else:
            row_number, X_id, user, location, image, tag, created, X__v, image1 = row
            readimg = 0
            name = image1[image1.find('Photos/')+len('Photos/'): image1.find('JPG')+3]
            img = where_to_read + name
            name = resized_path + name
            print("name: ", name)
            print(os.path.isfile(name))
            if os.path.isfile(name):
                continue
            if os.path.isfile(img):
                print("is there")
                readimg = Image.open(img)
                width, height = readimg.size
                if  width <= height:
                    width_percent = new_size / float(width)
                    new_height = int(float(height) * float(width_percent))
                    new_width = new_size
                    readimg = readimg.resize((new_width, new_height), PIL.Image.ANTIALIAS)
                elif height <= width:
                    height_percent = new_size /float(height)
                    new_width = int(float(width) * float(height_percent))
                    new_height = new_size
                    readimg = readimg.resize((new_width, new_height), PIL.Image.ANTIALIAS)

                readimg.save(name)
'''

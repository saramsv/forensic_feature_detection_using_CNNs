#This script nly extracts lines with a specific tag.

import csv
import sys
import os.path
import PIL
from PIL import Image

if __name__ == '__main__':
    csv_file = sys.argv[1]#normalized512by512_tags.csv
    extracted_tags_file = sys.argv[2] #scav_normalized512by512.csv
    #extracted_tag = ['mold', 'purge', 'scavenging']#sys.argv[3]
    extracted_tag = ['scavenging']#sys.argv[3]

    f = open(csv_file)
    data = csv.reader(f)
    cropped_imgs_metedata = []
    cropped_imgs_metedata.append(['row_number', 'X_id', 'user', 'location', 'image', 'tag', 'created', 'X__v', 'image1'])
    num = 0
    counter = 0
    for line in data:
        counter += 1
        if counter <  2:
            continue 
        else:
            row_number, X_id, user, location, image, tag, created, X__v, image1 = line
            if tag in extracted_tag:
                num +=1
                cropped_imgs_metedata.append(line)
    print("Number of extracted lines", num)
    #complement of multiclasses
    with open (extracted_tags_file,'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(cropped_imgs_metedata)

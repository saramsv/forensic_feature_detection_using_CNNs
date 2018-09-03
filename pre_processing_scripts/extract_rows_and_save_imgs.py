#!/usr/bin/env python3
#This is for testing images that are not squared. So the csv is going to be the same as the original ones. Thus, I am going to only extract lines with a specific tag.
# ./extract_rows_and_save_imgs.py normalized512by512_tags.csv scav_512by512.csv resized512_512_all_imgs/ scav_imgs_512by512/ scavenging
import csv
import sys
import os.path
import cv2

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("usage: csv_file dest_csv src_dir dest_dir tag")
        exit()
    csv_file = sys.argv[1]#normalized512by512_tags.csv
    extracted_tags_file = sys.argv[2] #scav_normalized512by512.csv
    #extracted_tag = ['mold', 'purge', 'scavenging']#sys.argv[3]
    src_dir = sys.argv[3]
    dest_dir = sys.argv[4]
    extracted_tag = sys.argv[5] #['scavenging']

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
                im = cv2.imread(src_dir + '/'+ image1)
                cv2.imwrite(dest_dir + '/' + image1, im)
                cropped_imgs_metedata.append(line)
    print("Number of extracted lines", num)
    #complement of multiclasses
    with open (extracted_tags_file,'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(cropped_imgs_metedata)

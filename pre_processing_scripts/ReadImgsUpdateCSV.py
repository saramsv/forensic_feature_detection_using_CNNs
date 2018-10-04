#!user/bin/env python3
#this script adds the first and last col
# replaces photo with photos
#converts all tags to lowercase
#splits the tags seperated by ;

#python3 ReadImgsUpdateCSV.py ../tags.csv.20181001 /icputrd/arf/mean.js/public ../AllLabeledImgs/MummImgs Mumm.20181001.correct.format.csv mummification

import csv
import glob
import os
import cv2
import sys 
def addFirstLastCol(row, i):
    row.insert(0, i-1)
    photo2photos(row)
    row.append(row[4][row[4].find("Photos/")+len("Photos/"):])
    return row[:] #because of copy by refrense

def photo2photos(row):
    row[4] = row[4].replace("Photo/","Photos/")
    return row[:]     

def lowercaseTag(row):
    row[5] = row[5].lower()
    return row[:] 

def splitTagRemoveVal(row):
    new_rows = []
    tags = row[5].split(";")
    for tag in tags:
        if "validate" not in tag:
            if tag in dic:
                tag = dic[tag.strip()]
            row[5] = tag
            new_rows.append(row[:])
        elif "validate" in tag and "yes" in tag:
            row[5] = tag.split()[1]
            new_rows.append(row[:])
        if "validate" in tag and "no" in tag:
            continue
    return new_rows

def read_img(path):
    flag = True
    if os.path.isfile(path) == False:
        print("this image does not exist:" , path)
        flag = False
    img_obj = cv2.imread(path, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR)
    return img_obj, flag



###############################

if len(sys.argv) < 6:
    print("Usage: script csv_read_from read_image_from store_image_at store_csv_at label")
    exit()

src_csv_file = sys.argv[1]
src_dir = sys.argv[2]
dest_dir = sys.argv[3]
dest_csv_file = sys.argv[4]
label = sys.argv[5]
dic_file = 'labelsdict.txt'

dic_f = open(dic_file)
dic_data = csv.reader(dic_f)
dic = {}
for d in dic_data:
    split_d = d[0].split(":")
    dic[split_d[0]] = split_d[1].strip()


rows = []
rows.append(['row_number','_id','user','location','image','tag','created','__v','image1'])
f = open(src_csv_file)
data = csv.reader(f)
i = 0
row_count = 0
for row in data:
    if i == 0:
        i += 1
        continue

    else:
        addFirstLastCol(row,i)
        lowercaseTag(row)

        img_path = os.path.join(src_dir , row[4][row[4].find("20"):]) 
        img_name = row[8]
        if label not in row[5]:
            continue

        img_obj, flag = read_img(img_path)
        if flag == False:
            continue
        img_name = dest_dir + '/' +  img_name
        cv2.imwrite(img_name, img_obj) 

        new_rows = splitTagRemoveVal(row)
        for r in new_rows:
            r[0] = row_count
            row_count += 1
            rows.append(r)
        i += 1
        
print(i)
with open(dest_csv_file, 'w') as output:
    writer = csv.writer(output, lineterminator = '\n')
    writer.writerows(rows)

#this is how I run this: python3 replace_tags.py dictionary_tags.txt ../tags/total_tags.csv

import csv
import json
import sys


filename = sys.argv[1]
tagfile = sys.argv[2]
rows = []
rows.append(['row_number', 'X_id', 'user', 'location', 'image', 'tag', 'created', 'X__v', 'image1'])
tag_dic = {}
with open(filename) as fp:
    fp.readline()
    reader = csv.reader(fp)
    for row in reader:
        row = row[0].split(":")
        if row[0] not in tag_dic:
            tag_dic[row[0]] = row[1].strip()

with open(tagfile) as fp2:
    fp2.readline()
    reader = csv.reader(fp2)
    for row in reader:
        tag_line = row[5]
        tags = tag_line.split(";")
        for tag in tags:
            tag = tag.lower().strip()
            if tag in tag_dic:
                row[5] = tag_dic[tag]
                rows.append(row[:])
'''
with open('total_replaced_tags','w') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(rows)
'''

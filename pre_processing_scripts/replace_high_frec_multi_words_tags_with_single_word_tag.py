#!/usr/bin/env python
## This script is used to find unique tags, and replace the single or multi words tags with a single word tag mentioned in 'classes'

import csv
import sys
import json

rows = []
classes = "eggs", "skin", "fly", "head", "mummification", "hair", "lividity", "duplicate", "discoloration", "adipocere", "nose", "delete", "foot", "ear", "larvae", "mold", "ai", "eye", "scavenging", "bone", "mouth", "purge", "marbling", "stake"

rows.append(['row_number', 'X_id', 'user', 'location', 'image', 'tag', 'created', 'X__v', 'image1'])
filename = sys.argv[1]
all_tags = {}
with open(filename) as fp:
    fp.readline()
    reader = csv.reader(fp)
    for row in reader: 
        tag_line = row[5]
        tags = tag_line.split(';')
        for tag in tags:
            tag = tag.lower().strip()
            if tag == "":
                continue
            tag = tag.split(" ")
            for i in tag:
                if i in classes:
                    row[5] =  i
                    rows.append(row)

with open('high_frec_single_word_tag2.csv','w') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(rows)

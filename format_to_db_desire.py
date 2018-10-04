# run : ython format_to_db_desire.py no_validated_tags.csv > formated_NOs.csv
#!/usr/bin/env python

import csv
import sys
import json

f = open(sys.argv[1])
data = csv.reader(f)
for row in data:
    dic = {}
    dic["_id"] = row[0]
    dic["user"] = row[1]
    dic["location"] = row[2]
    dic["image"] = row[3]
    dic["tag"] = row[4]
    dic["created"] = row[5]
    dic["__v"] = int(row[6])
    print (json.dumps(dic))

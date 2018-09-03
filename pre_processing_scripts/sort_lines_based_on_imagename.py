#!/usr/bin/env python
## This script is used to cound unique tags and also seperate the tag info for
##specific tags in a seperated csv file

import csv
import sys
import json
import pandas as pd

if len(sys.argv) < 2 :
    print("usage: python script csv_file")
    exit()
filename = sys.argv[1]
df_res = pd.DataFrame()
df = pd.read_csv(filename)
for i in range(df['image1'].unique().shape[0]):
    all_sub_imgs = df.loc[df['image1'] == df['image1'].unique()[i]]
    if all_sub_imgs.shape[0] > 1:
        print(all_sub_imgs)
    df_res = df_res.append(all_sub_imgs)
df_res.to_csv('sorted_mumm.csv')

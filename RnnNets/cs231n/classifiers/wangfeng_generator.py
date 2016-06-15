#-*- coding:utf-8 -*-
word_to_index = {}

file_path = "/Users/stgy/wangfeng.txt"
with open(file_path) as f:
    lines = f.readlines()
    for line in lines:
        if line != '\n':
            print line[2]


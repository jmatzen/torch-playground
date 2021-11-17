import csv
import math
import numpy as np
import sys

DAYS = 10
SKIP = 11

def normalize_col(rows,n):
    min = 99999999999
    max = 0
    for r in rows:
        min = r[n] if r[n] < min else min
        max = r[n] if r[n] > max else max
    for r in rows:
        r[n] = ((r[n]-min) / (max-min)) * 2.0 - 1.0
    return (min,max)
    

def normalize(rows):
    scale=[]
    scale.append(normalize_col(rows,0))
    scale.append(normalize_col(rows,1))
    scale.append(normalize_col(rows,2))
    scale.append(normalize_col(rows,3))
    scale.append(normalize_col(rows,4))
    return scale

rows = []
with open(f'{sys.argv[1]}.csv') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    next(reader,None)
    for row in reader:
        row = list(map(lambda x: float(x), 
            (row[1],row[2],row[3],row[4],row[6])))
        rows.append(row)



for i in range(len(rows)-SKIP):
    sample = []
    for row in range(i,i+DAYS):
        sample.append(rows[row].copy())
    sample.append(rows[i+SKIP].copy())
    
    scales = normalize(sample)
    result = {}
    result['target'] = sample[-1:][0]
    result['data'] = sample[:-1]
    result['scaling'] = scales
    print(result)



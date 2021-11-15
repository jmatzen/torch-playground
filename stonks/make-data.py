import csv
import math

DAYS = 16
SKIP = 19

def normalize_col(rows,n):
    min = 99999999999
    max = 0
    for r in rows:
        min = r[n] if r[n] < min else min
        max = r[n] if r[n] > max else max
    for r in rows:
        r[n] = ((r[n]-min) / (max-min))
    

def normalize(rows):
    normalize_col(rows,0)
    normalize_col(rows,1)
    normalize_col(rows,2)
    normalize_col(rows,3)
    normalize_col(rows,4)
    normalize_col(rows,5)

def tonum(rows):
    rowsout=[]
    for row in rows:
        col=[]
        for val in row:
            col.append(float(val))
        rowsout.append(col)
    return rowsout

rows = []
with open('spy.csv') as csvfile:
    reader = csv.reader(csvfile,delimiter=',')
    next(reader,None)
    for row in reader:
        rows.append(row[1:])

rows = tonum(rows)

for i in range(len(rows)-SKIP):
    sample = []
    for row in range(i,i+DAYS):
        sample.append(rows[row].copy())
    sample.append(rows[i+SKIP].copy())
    normalize(sample)
    last = sample[-1:]
    sample = sample[:-1]
    delta = last[0][0]-sample[-1:][0][0]
    delta = delta + 1
    print((sample,delta))



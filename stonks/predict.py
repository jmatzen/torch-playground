#!/usr/bin/env python
import torch
import numpy as np
import random
from torch import nn, Tensor
import math
from stonks import *
import requests
import time
import csv
from pprint import pprint
from sys import argv
device = "cpu"# "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = torch.load("model.bin")
model.eval()


def predict(sym):
    r = requests.get(f'https://query1.finance.yahoo.com/v7/finance/download/{sym}',
        params={
            'period1': math.floor(time.time()-86400*20),
            'period2': math.floor(time.time()),
            'interval': '1d',
            'events': 'history',
    #        'includeAdjustedClose': 'false'
        },
        headers={
            'user-agent': 'yup/1.0'
        })
    lines = r.text.split('\n')
    reader = csv.reader(lines,delimiter=',')
    next(reader,None)
    rows = []
    for row in reader:
        row = list(map(lambda x: float(x), 
            (row[1],row[2],row[3],row[4],row[6])))
        rows.append(row)
    rows = rows[-11:]
    scale = normalize(rows)

    data = rows[:10]
    actual = rows[-1:]

    def scale_(val, scaling):
        result = val.copy()
        for i in range(len(val)):
            base = scaling[i][0]
            delta = scaling[i][1] - base
            result[i] = (val[i] * 0.5 + 0.5) * delta + base
        return result


    data = torch.tensor(data).to(device)
    predict = model(data)

    actual = scale_(actual[0], scale)
    predict = scale_(predict.tolist(), scale)
    actual_gain = actual[3] - actual[0]
    predict_gain = predict[3] - predict[0]
    win = "win" if actual_gain*predict_gain>0 else "loss"

    print(f"{sym} " \
        f" actual={actual[3]} {actual[0]} {actual_gain:>0.2f}" \
        f" predicted={predict_gain:>0.2f} {win}")



syms = """tfc su iefa zip geo bkln cnq
 on snow cvs net gm rblx meta
 eww nrz gis aple vcit mrvl efa ttd srln
 jnk gld mchp usmv pff vgk schf ea
 peg cnx fast pbct ief qcom xly ccj eqh
 cg cnc info seah iyr vxus dell eqt igsb
 mpw bx nke fez crm hta wmt vea asan amd
 cost low schp hd ewt pgx stx csco""".replace('\n','').split(' ')

for sym in syms if len(argv)==1 else argv[1:]:
    predict(sym)
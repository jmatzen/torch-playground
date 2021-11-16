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
            'period1': math.floor(time.time()-86400*15),
            'period2': 1636934400,
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
    rows = rows[-6:]
    scale = normalize(rows)
    # print("---- scale factors")
    # pprint(scale)

    data = rows[:5]
    actual = rows[-1:]
    # print("---- data")
    # pprint(data)
    # print("---- actual")
    # pprint(actual)

    def scale_(val, scaling):
        return (val * 0.5 + 0.5) * (scaling[1] - scaling[0]) + scaling[0]

    prevclose = scale_(data[4][3], scale[3])

    actual = scale_(actual[0][3], scale[3])
    

    data = torch.tensor(data).to(device)

    predict = model(data)
    predict = scale_(predict.tolist()[3], scale[3])

    result = ""
    if (actual > prevclose and predict > prevclose) or (actual < prevclose and predict < prevclose):
        result = "win"
    else:
        result = "loss"
    print(f"{sym} prev={prevclose:>0.2f}" \
        f" actual={actual:>0.2f}" \
        f" predicted={predict:>0.2f} ({result})")



syms = """tfc su iefa zip geo bkln cnq
 on snow cvs net gm rblx meta
 eww nrz gis aple vcit mrvl efa ttd srln
 jnk gld mchp usmv pff vgk schf ea
 peg cnx fast pbct ief qcom xly ccj eqh
 cg cnc info seah iyr vxus dell eqt igsb
 mpw bx nke fez crm hta wmt vea asan amd
 cost low schp hd ewt pgx stx csco""".replace('\n','').split(' ')

for sym in syms:
    predict(sym)
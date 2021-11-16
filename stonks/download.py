# https://query1.finance.yahoo.com/v7/finance/download/QQQ?period1=1605487748&period2=1637023748&interval=1d&events=history&includeAdjustedClose=true
# https://query1.finance.yahoo.com/v7/finance/download/QQQ?period1=921024000&period2=1636934400&interval=1d&events=history&includeAdjustedClose=true
import requests

r = requests.get('https://query1.finance.yahoo.com/v7/finance/download/QQQ',
    params={
        'period1': 921024000,
        'period2': 1636934400,
        'interval': '1d',
        'events': 'history',
#        'includeAdjustedClose': 'false'
    },
    headers={
        'user-agent': 'yup/1.0'
    })
print(r.text)

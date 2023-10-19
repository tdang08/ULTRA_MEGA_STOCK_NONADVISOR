import pandas
import json
import requests
import datetime
import pandas as pd
from constants import apikey

apikey = apikey
website = "https://api.twelvedata.com"
request_type = "/time_series?"
parameters  = {"symbol": "msft",
               "interval": "1day",
               "apikey": apikey,
               "outputsize":"100"} # Size can be 1-5000

request = website + request_type

# GET JSON FROM REST API
response = requests.get(url=request, params=parameters)
obj = json.loads(response.content)

# USE JSON MOCK DATA
# json_dump = '{"meta": {"symbol": "AAPL", "interval": "1day", "currency": "USD", "exchange_timezone": "America/New_York", "exchange": "NASDAQ", "mic_code": "XNGS", "type": "Common Stock"}, "values": [{"datetime": "2023-10-13", "open": "181.42000", "high": "181.92999", "low": "178.14000", "close": "178.85001", "volume": "51427100"}, {"datetime": "2023-10-12", "open": "180.07001", "high": "182.34000", "low": "179.03999", "close": "180.71001", "volume": "56743100"}, {"datetime": "2023-10-11", "open": "178.20000", "high": "179.85001", "low": "177.60001", "close": "179.80000", "volume": "47551100"}, {"datetime": "2023-10-10", "open": "178.10001", "high": "179.72000", "low": "177.95000", "close": "178.39000", "volume": "43698000"}, {"datetime": "2023-10-09", "open": "176.81000", "high": "179.05000", "low": "175.80000", "close": "178.99001", "volume": "42390800"}], "status": "ok"}'
# obj = json.loads(json_dump)

data = {'date':[], 'close':[], 'volume':[]}
for val in obj['values']:
    data['date'].insert(0,datetime.datetime.strptime(val['datetime'], '%Y-%m-%d'))
    data['close'].insert(0,float(val['close']))
    data['volume'].insert(0,float(val['volume']))

# GET DATAFRAME FROM JSON
dataframe = pd.DataFrame(data=data)
# GET DATAFRAME FROM CSV
# dataframe = pd.read_csv('output100.csv')


labels = []
index_count = 0
previous_row = 0
dataframe = dataframe.set_index('date')

# calculate label for closing price change from day to day
for current_row in dataframe.itertuples():
    if index_count > 0:
        # define label according to price change
        # If previous day is almost same as current day (disregard decimals), 
        #       set label as 0
        # If previous day is lower than current day, set label as 1
        # If previous day is higher than current day, set label as -1
        if (float(previous_row[2]) == float(current_row[2])): labels.insert(0,0)
        elif (float(previous_row[2]) < float(current_row[2])): labels.append(1)
        elif (float(previous_row[2]) > float(current_row[2])): labels.append(-1)
    else: labels.append(0)
    index_count = index_count + 1
    previous_row = current_row
if 'label' in dataframe.columns:
    dataframe['label'] = labels
else: dataframe.insert(2, "label", labels, True)
print(dataframe)

# Calculate moving average from 20 days
# Price is closing price
n_days_price_buffer = []
N_DAYS = 20
index_count = 0
sma_buffer = []
std_dev_buffer = []
for current_row in dataframe.itertuples():
    if index_count > 19:
        n_days_price_buffer.insert(0,float(current_row[2]))
        n_days_price_buffer.pop()
        avg = sum(n_days_price_buffer)/N_DAYS
        variance = 0

        # Std dev calculation
        # Get positive difference from mean
        for val in n_days_price_buffer:
            variance = variance + (val - avg)**2
        # Average difference
        variance = variance/N_DAYS
        # undo squared values in difference calculation
        variance = variance**0.5

        sma_buffer.append(avg)
        std_dev_buffer.append(variance)
        index_count = index_count + 1
    else:
        index_count = index_count + 1
        n_days_price_buffer.insert(0,float(current_row[2]))
        sma_buffer.append(0)
        std_dev_buffer.append(0)

# Calculate upper and lower bollinger band
# bol = mean +/- num_of_std_dev * std_dev
bollinger_upper = []
bollinger_lower = []
NUM_OF_STD_DEV = 2
for i in range(len(sma_buffer)):
    if i < 19:
        bollinger_lower.append(0)
        bollinger_upper.append(0)
    else:
        bollinger_upper.append(sma_buffer[i] + NUM_OF_STD_DEV * std_dev_buffer[i])
        bollinger_lower.append(sma_buffer[i] - NUM_OF_STD_DEV * std_dev_buffer[i])

closing_prices = dataframe['close'].to_numpy()
bollinger_band_norm = []

# Normalize bollinger band values to one vector
# If < 0, price is below band
# If > 1, price is above band
for i in range(len(closing_prices)):
    if i > 19:
        norm = (closing_prices[i] - bollinger_lower[i])
        norm = norm/(bollinger_upper[i]-bollinger_lower[i])
        bollinger_band_norm.append(norm)
    else:
        bollinger_band_norm.append(0)

dataframe.insert(3,'bb_normalized', bollinger_band_norm)


# Calculate on-balance volume
# This will tell us how much volume has accumulated for each closing day
# If we see the slope as really high then, it's an indication that price will experience steep change
# If slope is negative then price will dip. but who knows...
current_balance = 0
previous_balance = 0

obv = []

for data in dataframe[['volume','close','label']].itertuples():
    index,volume,close,label = data
    if label == 0: current_balance = previous_balance + 0
    elif label == 1: current_balance = previous_balance + volume
    elif label == -1: current_balance = previous_balance - volume
    previous_balance = current_balance
    # Normalize by using current volume as base
    # The numbers get ridiculous!!
    obv.append(current_balance/volume)
dataframe['obv_norm'] = obv

dataframe.to_csv('msft_test.csv')


# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(random_state=0)
# # 22 onwards to ignore the initial data where moving average has yet to form
# X = dataframe[['obv_norm', 'bb_normalized']][22:].to_numpy()
# y = dataframe[['label']][22:].to_numpy()
# clf.fit(X,y)

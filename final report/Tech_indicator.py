# %%
import pandas as pd
import requests
import time
# %%
def get_date_list(start_year, end_year):
    date_list = []
    for year in range(start_year, end_year+1):
        for month in range(1, 13):
            if month < 10:
                date_list.append(str(year) + '0' + str(month) + '01')
            else:
                date_list.append(str(year) + str(month) + '01')
    return date_list
            
date_list = get_date_list(2015, 2020)
for index, date in enumerate(date_list):
    print(index)
    url = f'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=html&date={date}&stockNo=2330'
    data = pd.read_html(requests.get(url).text)[0]
    data.columns = data.columns.droplevel(0)
    if index == 0:
        df_2330 = pd.DataFrame(data)
    else:
        df_2330 = pd.concat([df_2330, data])
    time.sleep(5)

# %%
for index, date in enumerate(date_list):
    print(index)
    url = f'https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=html&date={date}&stockNo=3008'
    data = pd.read_html(requests.get(url).text)[0]
    data.columns = data.columns.droplevel(0)
    if index == 0:
        df_3008 = pd.DataFrame(data)
    else:
        df_3008 = pd.concat([df_3008, data])
    time.sleep(5)

# %%
mapper = {'日期':'date', '成交股數':'volume', '成交金額': 'tradingvolume', '開盤價':'open', '最高價':'high', '最低價':'low', '收盤價':'close', '成交筆數':'number_trade', '漲跌價差': 'price_difference'}

df_2330 = df_2330.rename(columns=mapper)
df_3008 = df_3008.rename(columns=mapper)
# %%
df_2330 = df_2330.drop(['tradingvolume', 'number_trade'], axis=1)
df_3008 = df_3008.drop(['tradingvolume', 'number_trade'], axis=1)
# %%
df_2330
# %%
from talib import abstract
# %%
MA = abstract.MA
KD = abstract.STOCH
MACD = abstract.MACD
RSI = abstract.RSI
STDDEV = abstract.STDDEV
# %%
def modify_df(df):
    df['ma10'] = MA(df, timeperiod=10)
    df['ma30'] = MA(df, timeperiod=30)
    macd = MACD(df)
    df['rsi'] = RSI(df)
    df = pd.concat([df,macd], axis=1)
    df['std'] = STDDEV(df)
    return df

df_2330 = modify_df(df_2330)
df_3008 = modify_df(df_3008)


# %%
import numpy as np
price_difference = np.array(df_2330['price_difference'], dtype=float)
close = np.array(df_2330['close'], dtype=float)
for index, value in enumerate(df_2330['price_difference']):
    if value == 'X0.00':
        price_difference[index] = close[index] - close[index - 1]
df_2330['price_difference'] = price_difference
percentage_2330 = [0] + list(price_difference[1:] / close[:-1])
df_2330['percentage'] = percentage_2330

price_difference = np.array(df_3008['price_difference'], dtype=float)
close = np.array(df_3008['close'], dtype=float)
for index, value in enumerate(df_3008['price_difference']):
    if value == 'X0.00':
        price_difference[index] = close[index] - close[index - 1]
df_3008['price_difference'] = price_difference

percentage_3008 = [0] + list(price_difference[1:] / close[:-1])
print(len(percentage_2330))
print(df_2330.shape)
df_3008['percentage'] = percentage_3008
# %%
import numpy as np
np.array(df_2330['price_difference'])
# %%
df_2330 = df_2330.iloc[33:,:]
# %%
df_3008 = df_3008.iloc[33:,:]
# %%
final_csv = pd.read_csv('/home/pj/Downloads/final.csv')
df_2330 = pd.read_csv('台積電.csv')
# %%
df_2330.to_csv('台積電.csv')
df_3008.to_csv('大立光.csv')
# %%
print(df_2330.index)
# %%

# %%

# %%

# %%

# %%
final_csv
# %%
df_2330.iloc[668:,:]
# %%
df_3008.iloc[668:, :]
# %%
final_2330_csv = pd.concat([df_2330.iloc[668:,], final_csv['snownlp'], final_csv['bert']], axis=1)
# %%
df_2330.iloc[668:,:]
# %%
final_csv['bert']
# %%
df_2330 = df_2330.drop('Unnamed: 0', axis=1)
# %%
tmp = []
for i in
# %%
final_csv
# %%
df_2330['date'] = df_2330['date'].apply(lambda x: str(1911 + int(x[:3])) + x[3:])
df_2330
# %%
import numpy as np
index_array = []
for i in range(df_2330.shape[0]):
    #print(type(np.array(df_2330['date'])[i]))
    if str(np.array(df_2330['date'])[i]) in np.array(final_csv['date']):
        index_array.append(i)

print(index_array)
print(type(final_csv['date'][0]))

# %%
final = pd.concat([df_2330, final_csv['bert'], final_csv['snownlp']], axis=1)
# %%
final
# %%
final.to_csv('台積電final.csv')
# %%

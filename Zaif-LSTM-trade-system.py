import datetime
import poloniex
import pandas as pd
from datetime import time
import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation,Flatten
from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
from zaifapi import ZaifPublicApi
from zaifapi import ZaifTradeApi
from decimal import (Decimal)
import json
from numpy import sqrt
import pprint
import traceback


in_out_neurons = 1
hidden_neurons = 900
length_of_sequences = 10

model = Sequential()
model.add(LSTM(hidden_neurons, batch_input_shape=(None, length_of_sequences, in_out_neurons), return_sequences=False))
#model.add(Flatten())
model.add(Dense(in_out_neurons))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="adam",)
early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)

zaif_keys_json = open('apikey-data.json', 'r')
zaif_keys = json.load(zaif_keys_json)
KEY = zaif_keys['Zaif']['key']
SECRET = zaif_keys['Zaif']['secret']

zaif_trade = ZaifTradeApi(KEY,SECRET)
last_trade_price=0
def moving_ave(last_price, period):
    sum_price = 0
    for price in last_price[-period:]:
        sum_price += price
    return sum_price / period


def st_div(price, period):
    sum_price = 0
    sum_price_2 = 0
    for price in last_price[-period:]:
        sum_price += price
        sum_price_2 += price ** 2
    return sqrt((period * sum_price_2 - sum_price ** 2) / period * (period - 1))


def cancel_flug(trade_result):
    global order_id, cancel_flug
    if (trade_result["order_id"] != 0):
        order_id = trade_result["order_id"]
        cancel_flug = 1
    else:
        print("■ 取引が完了しました。")


def _load_data(data, n_prev=10):
    docX, docY = [], []
    for i in range(len(data) - n_prev):
        docX.append(data[i:i + n_prev])
        docY.append(data[i + n_prev])
    alsX = np.array(docX)
    alsY = np.array(docY)
    return alsX, alsY


def train_test_split(df, test_size=0.1, n_prev=20):
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)
    return (X_train, y_train), (X_test, y_test)


def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x - min) / (max - min)
    return result

def bitbuy(funds_jpy):
    try:
        per=1.0
        value=int(zaif_public.last_price('btc_jpy')["last_price"])-500
        bid_amount = (Decimal(funds_jpy)*Decimal(per) / value).quantize(Decimal('0.0001'))
        trade_result = zaif_trade.trade(currency_pair="btc_jpy", action="bid", price=value, amount=bid_amount)
        #last_trade_price_pre = last_trade_price
        #last_trade_price = last_price[-1]
        #not_able_bid_count = 0
        print('■ Bitcoinの購入申請を行いました')
        pprint.pprint(trade_result)
        print("購入注文価格: " + str(value))
        print("購入注文量　: " + str(bid_amount))
        if (trade_result["order_id"] == 0):
            print("■ ビットコインの購入取引が完了しました。")
    except:
        print("btc購入エラー:cant trade[bid]")
        traceback.print_exc()
def bitsell(funds_btc):
    try:
        # 売却
        ask_amount = funds_btc
        value=int(zaif_public.last_price('btc_jpy')["last_price"])
        trade_result = zaif_trade.trade(currency_pair="btc_jpy", action="ask", price=value, amount=ask_amount)
        print('■ Bitcoinの売却申請を行いました。')
        pprint.pprint(trade_result)
        print("売却注文価格: " + str(value))
        print("売却注文量　: " + str(ask_amount))
        if (trade_result["order_id"] == 0):
            print("■ ビットコインの売却取引が完了しました。")
    except:
        print("btc売却エラー:cant trade[ask]")
        traceback.print_exc()

last_price = []
price = 0
CANCEL_FLUG = False
period = 20
low_borin = 0
m_ave = 99999999
high_borin = 99999999
t20 = 0  # 長期戦
t5 = 0  # 短期戦
bought = 0
balance = 10000  # 残高
buybitcount = 0
# y_train[-20:])
mss = MinMaxScaler()
indexmax = -1
indexmin = 30
count = 0
no_elpse=1
last_price = []
price = 0
CANCEL_FLUG = False
period = 20
low_borin = 0
m_ave = 99999999
high_borin = 99999999
t20 = 0  # 長期戦
t5 = 0  # 短期戦
bought = 0
balance = 10000  # 残高
buybitcount = 0
# y_train[-20:])
mss = MinMaxScaler()
indexmax = -1
indexmin = 300
count = 0
over_minutes = 0
over_len = 0

while (True):

    start_time = time.time()
    zaif_public = ZaifPublicApi()
    print("■ 現在の情報です")
    try:
        funds_btc = Decimal(zaif_trade.get_info2()['funds']['btc']).quantize(Decimal('0.0001'))
        funds_jpy = zaif_trade.get_info2()['funds']['jpy']
        price = int(zaif_public.last_price('btc_jpy')["last_price"])
        last_price.append(price)

    except:
        print("エラー:Cant get data 30秒待ちます")
        time.sleep(30)
        continue

    finally:
        print("市場取引価格: " + str(last_price[-1]))
        print("btc資産: " + str(funds_btc))
        print("jpy資産: " + str(funds_jpy))
        print("最終取引価格: " + str(last_trade_price))

    if indexmin == count or indexmax == count:
        if len(last_price) >= 300:  # 60:
            #last = np.array(last_price)
            if over_minutes == 1:
                last = np.array(last_price[-over_len:])
            else:
                last = np.array(last_price)

            last = min_max(last)
            X_train, y_train = _load_data(last)
            a, b = X_train.shape
            X_train = X_train.reshape(a, b, 1)
            # 予測開始
            history = model.fit(X_train, y_train, batch_size=600, epochs=10, validation_split=0.1,
                                callbacks=[early_stopping])
            future_test = y_train[-10:]
            time_length = len(future_test)
            future_result = np.empty((1))
            future_result = np.delete(future_result, 0)
            for step2 in range(20):
                test_data = np.reshape(future_test, (1, time_length, 1))
                batch_predict = model.predict(test_data)

                future_test = np.delete(future_test, 0)
                future_test = np.append(future_test, batch_predict)

                future_result = np.append(future_result, batch_predict)

            #plt.plot(future_result)
            #plt.show()
            maxbit = future_result.max()
            minbit = future_result.min()
            indexmin = np.argmin(future_result)
            indexmax = np.argmax(future_result)
            print(str(indexmax + 1) + "分後:" + str(maxbit) + "で最大")
            print(str(indexmin + 1) + "分後:" + str(minbit) + "で最小")

            count = 0

            if indexmin == 0:
                if funds_jpy>5000:  #bought == 0:5000円以上持ってたら買い
                    print("-------------------------今から買います--------------------------------")
                    bitbuy(funds_jpy)
                    #buybitcount = balance / price
                    print(str(balance) + "円で買いました")
                    #balance = 0
                    #bought = 1
            if indexmax == 0:
                if funds_btc>0.0001:#bought == 1:
                    print("ーーーーーーーーーーーーーー今から売りますーーーーーーーーーーーーーーーーーー")
                    bitsell(funds_btc)
                    #balance = price * buybitcount
                    #buybitcount = 0
                    bought = 0
                    print(str(balance) + "円で売りました")

    # t20=moving_ave(last_price,20)
    # t5=moving_ave(last_price,5)
    end_time = time.time()
    elpsed_time = end_time - start_time
    count = count + 1
    if no_elpse==1:
        if elpsed_time > 30:
            elpsed_time %= 30
            over_minutes = 1
            no_elpse=0
            over_len = len(last_price)
            print("take time over 30sec")

    print("かかった時間は" + str(elpsed_time) + "秒")
    if elpsed_time<60:
        time.sleep(60 - elpsed_time)
    else:
        time.sleep(30)

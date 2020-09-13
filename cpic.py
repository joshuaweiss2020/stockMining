import tushare as ts

import tushare.stock.indictor as indx


import pandas as pd
import numpy as np

def cci(df, n=14):
    """
    顺势指标
    TYP:=(HIGH+LOW+CLOSE)/3
    CCI:=(TYP-MA(TYP,N))/(0.015×AVEDEV(TYP,N))
    """
    _cci = pd.DataFrame()
    _cci["trade_date"] = df['trade_date']
    typ = (df.high + df.low + df.close) / 3
    _cci['cci'] = ((typ - typ.rolling(n).mean()) /
                   (0.015 * typ.rolling(min_periods=1, center=False, window=n).apply(
                    lambda x: np.fabs(x - x.mean()).mean())))
    return _cci

# ts.get_hist_data('601601',start='2020-08-15',end='2020-09-01')

# token = '3be8423f505c5683743fcfc7ef9083a222e965161d3f1832f10fa9cc'
#
# ts.set_token(token)
#
#
# pro = ts.pro_api()
#
# data = pro.weekly(ts_code="601601.SH",start_date='20200801',end_date='20200831')
# print(data)
# print(indx.kdj(data))
# #l=indx.rsi(data)
a,b,c = indx.macd(data)
# print("MACD:","OSC:",a," DIFF:",b,"DEM:",c)
#
# print(cci(data))
#
# df=pro.daily(ts_code='601601.SH', start_date='20200801', end_date='2020901') #fields="trade_date,open,close,vol,amount"
# # df = pro.trade_cal(exchange='', start_date='20200801', end_date='20200901', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')
# print(df.columns)

# help(ts)

data = pd.Series(range(10))

data_r = data.rolling(3,min_periods=1).sum()
data_r1 = data.rolling(3,min_periods=2).sum()
data_r2 = data.rolling(3).sum()
print(data)
print(data_r)
print(data_r1)
print(data_r2)
data = pd.DataFrame(data)
print(pd.Series.rolling(data,3))

# print(help(pd.Series.rolling))
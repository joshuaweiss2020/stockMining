from ctypes import c_longdouble

import tushare as ts
import pandas as pd
import numpy as np

def ema(data, n=12, val_name="close",debug=False):
    import numpy as np
    '''
        指数平均数指标 Exponential Moving Average
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          n:int
                      移动平均线时长，时间单位根据data决定
          val_name:string
                      计算哪一列的列名，默认为 close 收盘值

        return
        -------
          EMA:numpy.ndarray<numpy.float64>
              指数平均数指标
    '''

    data = data.sort_values(ascending=True, by=["trade_date"], inplace=False) #原数据的index是按最新日期index=0
    data = data.reset_index()
    prices = []

    EMA = []

    past_ema = 0
    for index, row in data.iterrows():
        if index == 0:
            past_ema = row[val_name]
            EMA.append(row[val_name])
        else:
            # Y=[2*X+(N-1)*Y’]/(N+1)
            today_ema = (2 * row[val_name] + (n - 1) * past_ema) / (n + 1)
            # today_ema = round(today_ema,2)
            past_ema = today_ema

            if debug:
                EMA.append(row["trade_date"] + ":" + str(today_ema))
            else:
                EMA.append(today_ema)

    return np.asarray(EMA)

def macd(data, quick_n=12, slow_n=26, dem_n=9, val_name="close"):
    import numpy as np
    '''
        指数平滑异同平均线(MACD: Moving Average Convergence Divergence)
        Parameters
        ------
          data:pandas.DataFrame
                      通过 get_h_data 取得的股票数据
          quick_n:int
                      DIFF差离值中快速移动天数
          slow_n:int
                      DIFF差离值中慢速移动天数
          dem_n:int
                      DEM讯号线的移动天数
          val_name:string
                      计算哪一列的列名，默认为 close 收盘值

        return
        -------
          MACD:numpy.ndarray<numpy.float64>
              MACD bar / OSC 差值柱形图 2*(DIFF - DEM)
          DIFF:numpy.ndarray<numpy.float64>
              差离值
          DEA:numpy.ndarray<numpy.float64>
              讯号线
    '''
    data = data.sort_values(ascending=True, by=["trade_date"], inplace=False) #原数据的index是按最新日期index=0
    data = data.reset_index()
    ema_quick = np.asarray(ema(data, quick_n, val_name))
    ema_slow = np.asarray(ema(data, slow_n, val_name))
    DIFF = ema_quick - ema_slow
    data["diff"] = DIFF
    DEA = ema(data, dem_n, "diff",debug=False)
    MACD = (DIFF - DEA ) * 2
    return MACD, DIFF, DEA
    # return data

def kdj(data):
    '''
        随机指标KDJ
        Parameters
        ------
          data:pandas.DataFrame
                通过 get_h_data 取得的股票数据
        return
        -------
          K:numpy.ndarray<numpy.float64>
              K线
          D:numpy.ndarray<numpy.float64>
              D线
          J:numpy.ndarray<numpy.float64>
              J线
    '''

    K, D, J = [], [], []
    last_k, last_d = None, None
    for index, row in data.iterrows():
        if last_k is None or last_d is None:
            last_k = 50
            last_d = 50

        c, l, h = row["close"], row["low"], row["high"]

        rsv = (c - l) / (h - l) * 100

        k = (2 / 3) * last_k + (1 / 3) * rsv
        d = (2 / 3) * last_d + (1 / 3) * k
        j = 3 * k - 2 * d

        K.append(k)
        D.append(d)
        J.append(j)

        last_k, last_d = k, d

    return np.asarray(K), np.asarray(D), np.asarray(J)

def LV(data,n):  #n天最小值
    return pd.Series.rolling(data,n).min()

def HV(data,n):  #n天最大值
    return pd.Series.rolling(data,n).max()

def sortA(data):
    data = data.sort_values(ascending=True, by=["trade_date"], inplace=False) #原数据的index是按最新日期index=0
    data = data.reset_index()
    return data

def SMA(DF, N, M):  #N日的移动平均，M为权重
    DF = DF.fillna(0)
    z = len(DF)
    var = np.zeros(z)
    var[0] = DF[0]
    for i in range(1, z):
        var[i] = (DF[i] * M + var[i - 1] * (N - M)) / N
    for i in range(z):
        DF[i] = var[i]
    return DF

def KDJ(data, N, M1, M2):
    C = data['close']
    H = data['high']
    L = data['low']
    RSV = (C - LV(L, N)) / (HV(H, N) - LV(L, N)) * 100
    K = SMA(RSV, M1, 1)
    D = SMA(K, M2, 1)
    J = 3 * K - 2 * D
    DICT = {'KDJ_K': K, 'KDJ_D': D, 'KDJ_J': J}
    VAR = pd.DataFrame(DICT)
    return VAR



def profit(data,n=1,val_name="close",debug=False):
    '''
        计算T+n天的收益率
    '''
    data = data.sort_values(ascending=True, by=["trade_date"], inplace=False) #原数据的index是按最新日期index=0
    data = data.reset_index()
    PROFIT_N = []
    profit_rate = 0
    for index, row in data.iterrows():
        val = row[val_name]
        if index == 0:
            pass
        else:
            profit_rate = (row[val_name] - past_val) / past_val

        if debug:
            PROFIT_N.append(row["trade_date"] + ":" + str(round(profit_rate,4)))
        else:
            PROFIT_N.append(round(profit_rate,4))


        past_val = val

    PROFIT_N.append(0) #最后一天
    return PROFIT_N


if __name__=='__main__':
    token = '3be8423f505c5683743fcfc7ef9083a222e965161d3f1832f10fa9cc'
    ts.set_token(token)
    pro = ts.pro_api()
    data = pro.daily(ts_code="601601.SH", start_date='20200501', end_date='20200931')
    data = sortA(data)
    # data = data.sort_values(ascending=True,by=["trade_date"],inplace=False)
    data = data[["ts_code", "trade_date", "close"]]
    # data = data.head(20)
    # data["new_idx"] = range(12)
    # data = data.reset_index()
    # data.set_index("new_idx")
    # print(data.columns)
    # print(data)
    # print(ema(data,n=20))
    # data = macd(data)
    # OSC, DIFF, DEM = macd(data)
    # print(OSC)
    # print(data[["trade_date","diff"]])
    # profits = profit(data,debug=True)
    # print(profits)
    print(data["close"].tail(20))
    lv = HV(data["close"],10)
    print(lv)
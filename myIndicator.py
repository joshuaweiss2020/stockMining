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

    # data = data.sort_values(ascending=True, by=["trade_date"], inplace=False) #原数据的index是按最新日期index=0
    # data = data.reset_index()
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

def MACD(data, quick_n=12, slow_n=26, dem_n=9, val_name="close"):

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
    data["MACD"] = MACD
    data["DIFF"] = DIFF
    data["DEA"] = DEA
    # return MACD, DIFF, DEA
    return data

def MAX(A, B):
    var = IF(A > B, A, B)
    return var


def MIN(A, B):
    var = IF(A < B, A, B)
    return var

def IF(COND, V1, V2):
    var = np.where(COND, V1, V2)
    for i in range(len(var)):
        V1[i] = var[i]
    return V1

def LV(data,n):  #n天最小值
    return pd.Series.rolling(data,n).min()

def HV(data,n):  #n天最大值
    return pd.Series.rolling(data,n).max()

def sortA(data):
    data = data.sort_values(ascending=True, by=["trade_date"], inplace=False) #原数据的index是按最新日期index=0
    data = data.reset_index()
    return data

def SMA(DF, N, M): #n日移动平均线 M为权重
    DF = DF.fillna(0)
    z = len(DF)
    var = np.zeros(z)
    var[0] = DF[0]
    for i in range(1, z):
        var[i] = (DF[i] * M + var[i - 1] * (N - M)) / N
    for i in range(z):
        DF[i] = var[i]
    return DF


def KDJ(data, N, M1=3, M2=3):
   # 随机指标KDJ
    C = data['close']
    H = data['high']
    L = data['low']
    RSV = (C - LV(L, N)) / (HV(H, N) - LV(L, N)) * 100
    K = SMA(RSV, M1, 1)
    D = SMA(K, M2, 1)
    J = 3 * K - 2 * D
    # DICT = {'KDJ_K': K, 'KDJ_D': D, 'KDJ_J': J}
    # VAR = pd.DataFrame(DICT)
    # VAR["trade_date"] = data["trade_date"]
    data["K"] = K
    data["D"] = D
    data["J"] = J
    return data

def RSI(DF, N1=6, N2=12, N3=14):  # 相对强弱指标RSI1:SMA(MAX(CLOSE-LC,0),N1,1)/SMA(ABS(CLOSE-LC),N1,1)*100;
    CLOSE = DF['close']
    LC = REF(CLOSE, 1)
    RSI = SMA(MAX(CLOSE - LC, 0), N1, 1) / SMA(abs(CLOSE - LC), N1, 1) * 100
    # RSI2 = SMA(MAX(CLOSE - LC, 0), N2, 1) / SMA(abs(CLOSE - LC), N2, 1) * 100
    # RSI3 = SMA(MAX(CLOSE - LC, 0), N3, 1) / SMA(abs(CLOSE - LC), N3, 1) * 100

    # DICT = {'RSI1': RSI1, 'RSI2': RSI2, 'RSI3': RSI3}
    # VAR = pd.DataFrame(DICT)
    DF['RSI'] = RSI
    return DF

def CCI(DF,N): #N日的 CCI（N日）=（TP－MA）÷MD÷0.015
    '''
    CCI（N日）=（TP－MA）÷MD÷0.015
    其中，TP=（最高价+最低价+收盘价）÷3
    MA=近N日TP的累计之和÷N
    MD=近N日|（MA－TP）|的累计之和÷N
    0.015为计算系数，N为计算周期
    '''
    C = DF['close']
    H = DF['high']
    L = DF['low']
    TP = (H + L + C)/3
    MA_V = MA(TP,N)

    MD_V = AVEDEV(TP,N)
    DF["MD"] = MD_V
    DF['CCI'] = (TP-MA_V) / MD_V / 0.015
    return DF

    # return DF

def MA(DF, N):
    return pd.Series.rolling(DF, N).mean()

def AVEDEV(DF,N): #到算术平均值的绝对偏差的平均值
   MEAN = pd.Series.rolling(DF, N).mean()
   df = pd.DataFrame({"DF":DF,"MEAN":MEAN,"MD":np.zeros(DF.count())})
   for index, row in df.iterrows():
    if not np.isnan(row["MEAN"]) :
            s = 0
            for i in range(N):
             s+= abs(df.loc[index-i,"DF"]-row["MEAN"])
            row["MD"] = s/N
   return  df["MD"]

def REF(DF, N): #与前面第N日的值
    var = DF.diff(N) #与前面第N日的差值
    var = DF - var
    return var


def ROI(data,n=1): #计算第n日相对于当前的受益率
    # dataROI = data.copy()
    data.set_index("trade_date",inplace=False)
    ROI = data.shift(-n)["close"]/data["close"]-1
    idx = pd.Series(range(data["trade_date"].count()))
    data["idx"] = idx
    data.set_index(idx,inplace=True)
    data["ROI"] = ROI
    return data

if __name__=='__main__':
    token = '3be8423f505c5683743fcfc7ef9083a222e965161d3f1832f10fa9cc'
    ts.set_token(token)
    pro = ts.pro_api()
    data = pro.daily(ts_code="601601.SH", start_date='20200125', end_date='20200931')
    data = sortA(data)
    # data00 = data.tail(1)
    # print(data00)
    # tp = (data00["high"] + data00["low"] + data00["close"])/3
    # print(tp)
    # data28=data.tail(42)
    # data28["tp"] = (data28["high"] + data28["low"] + data28["close"])/3
    # print(data28)
    # # MA = data28.iloc[]["tp"].sum()/14
    #
    # print(data28[-14:]["tp"].sum()/14)
    # print("_______________________________")
    # print(data28[-14:]["tp"])
    # print("_______________________________")
    # data28["ma"] = data28["tp"].rolling(14).mean()
    # # data28["md"] = abs(data28["ma"]-data28["tp"]).rolling(14).mean()
    # data28["md"] = AVEDEV(data28["tp"],14)
    # data28["CCI"] = (data28["tp"]-data28["ma"])/data28["md"]/0.015
    # # data28["mf"] = data28["md"].rolling(14).mean().
    #
    # data100 = data28[["trade_date","tp","ma","md","CCI"]].dropna().tail(20)
    # # data100["ths"] = [-108.1,-27.68,-70.93,-26.07,-33.22]
    # # data100["xa"] = data100["ths"] - data100["CCI"]
    # # data100["shang"] = data100["ths"] / data100["CCI"]
    # print(data100)
    # data = data.sort_values(ascending=True,by=["trade_date"],inplace=False)
    # data = data[["ts_code", "trade_date", "close"]]
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
    # print(data["close"].tail(20))
    # hv = HV(data["close"],10)
    # print(lv)
    # kdj = KDJ(data,9)
    # print(kdj)
    # data.set_index(pd.datetime(data["trade_date"]),inplace=True)
    # data.drop("index")
  # print(data.trade_date)
  #   roi = ROI(data,5)
  #   print(roi*100)
  #   data.set_index("trade_date",inplace=True)
  #   print(data.shift(1)["close"],data["close"])
  #   rsi = RSI(data,6,12,24)
  #   print(rsi)
    data = ROI(data)
    data = CCI(data,14)
    data = RSI(data,6,12,24)
    data = KDJ(data,6,12,24)
    data = MACD(data)

    # print(cci[["trade_date","cci"]].tail(14))
    # print(rsi[["trade_date","RSI"]].tail(14))
    # print(kdj[["trade_date","K","D","J"]].tail(14))
    #
    # print(macd[["trade_date","MACD"]].tail(14))

    # data["CCI"] = cci["cci"]
    # data["RSI"] = rsi["RSI"]
    # data["K"],data["D"],data["J"] = kdj["K"] ,kdj["D"],kdj["J"]
    # data["MACD"] = macd["MACD"]
    # data["ROI"] = roi["ROI"]
    print(data[["trade_date","MACD","K","D","J","RSI","CCI","ROI"]].tail(30))


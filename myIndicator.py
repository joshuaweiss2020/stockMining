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

if __name__=='__main__':
    token = '3be8423f505c5683743fcfc7ef9083a222e965161d3f1832f10fa9cc'
    ts.set_token(token)
    pro = ts.pro_api()
    data = pro.daily(ts_code="601601.SH", start_date='20200501', end_date='20200931')

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
    OSC, DIFF, DEM = macd(data)
    print(OSC)
    # print(data[["trade_date","diff"]])
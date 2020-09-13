import tushare as ts
import pandas as pd
import numpy as np
import myIndicator as idx

def prepareIndicators(data): #计算各项指标
    data = idx.sortA(data)
    data = idx.ROI(data,3)
    data = idx.CCI(data,14)
    data = idx.RSI(data,6,12,24)
    data = idx.KDJ(data,9,3,3)
    data = idx.MACD(data,12,26,9)
    return data

def rule_KDJ_buy(data):
    '''
        j>80且不再创新高 sell
        j<20且不再创新低 buy
    '''
    # data = data[(data["J"]<20) & (data["J"]>idx.LV(data["J"],9))]
    data = data[(data["J"] < 20) ]
    print("J<20:",data["J"].count()) #670
    # data = data[(data["J"] > idx.LV(data["J"], 9))]
    # print("J<20 &J>LV9:",data["J"].count()) #508
    # data =data[data["ROI"]>0] #266
    # print(data[["trade_date","J","ROI"]])
    return data

def rule_RSI_buy(data):
    '''
        20<RSI<80且不再创新高 sell
        0<RSI<20且不再创新低 buy
    '''
    data = data[(data["RSI"]<20) & (data["RSI"]>0)]
    print("0<RSI<20",data["RSI"].count()) #110
    # data = data[(data["RSI"]>idx.LV(data["RSI"],6))]
    # print("0<RSI<20 & RSI>LV:",data["RSI"].count()) #110
    return data

def rule_CCI_buy(data):
    '''
        CCI>100且不再创新高 sell
        CCI<-100且不再创新低 buy
    '''
    data = data[(data["CCI"]<-100)]
    print("CCI<-100",data["CCI"].count()) #110
    data = data[(data["CCI"]>idx.LV(data["CCI"],14))]
    print("CCI<-100 & CCI>LV14:",data["CCI"].count()) #110
    return data

def rule_MACD_buy(data):
    '''
        MACD不再创新高 sell
        MACD不再创新低 buy
    '''
    data = data[(data["MACD"]<0)]
    print("MACD<0",data["MACD"].count()) #110
    data = data[(data["MACD"]>idx.LV(data["MACD"],9))]
    print("MACD>LV9:",data["MACD"].count()) #110
    return data


if __name__=='__main__':
    code='601601'
    token = '3be8423f505c5683743fcfc7ef9083a222e965161d3f1832f10fa9cc'
    ts.set_token(token)
    pro = ts.pro_api()
    data0 = pro.daily(ts_code=code+".SH", start_date='19810101', end_date='20200931')
    data0.to_json(code+"_all.json")
    data = pd.read_json(code+"_all.json")
    data = prepareIndicators(data)
    data = data.dropna()
    # print(data[["trade_date", "MACD", "K", "D", "J", "RSI", "CCI", "ROI"]].tail(30))
    data = rule_MACD_buy(data)
    data = rule_KDJ_buy(data)
    data = rule_RSI_buy(data)
    data = rule_CCI_buy(data)


    data.to_json(code+"_rule.json")
    data =data[data["ROI"]>0] #50
    # print("ROI>0:",data[["trade_date","RSI","ROI"]].count())
    print("ROI>0:",data[["ROI"]].count())

    data = pd.read_json(code+"_rule.json")
    print(data[["trade_date","close","J","RSI","CCI","MACD","ROI"]])

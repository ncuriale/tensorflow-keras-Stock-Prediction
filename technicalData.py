import numpy as np
import pandas as pd
import talib

def calcFeatures(df,indx):
    
    df1 = pd.DataFrame({'A' : []})

    #for i in range(0,7):
    for i in range(len(indx)):
        
        if (indx[i] and i==0):
            s1 = df['Open'] 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==1):
            s1 = df['High'] 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==2):
            s1 = df['Low'] 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==3):
            s1 = df['Close'] 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==4):
            s1 = df['Adj Close'] 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==5):
            s1 = df['Volume'] 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==6):
            s1 = pd.DataFrame( {'RSI': talib.RSI(np.array(df['Close'].values),timeperiod=14) }) 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==7):
            s1 = pd.DataFrame( {'ROC': talib.ROC(np.array(df['Close'].values), timeperiod=10) }) 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==8):
            s1 = pd.DataFrame( {'BETA': talib.BETA(np.array(df['High'].values), \
                np.array(df['Low'].values), timeperiod=5) }) 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==9):
            s1 = pd.DataFrame( {'STDDEV': talib.STDDEV(np.array(df['Close'].values), \
                timeperiod=5, nbdev=1) }) 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==10):
            s1 = pd.DataFrame( {'WILLR': talib.WILLR(np.array(df['Open'].values), \
                np.array(df['High'].values),np.array(df['Close'].values),timeperiod=40) }) 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==11):
            s1 = pd.DataFrame( {'SMA1': talib.SMA(np.array(df['Close'].values), timeperiod=30) }) 
            df1 = pd.concat([df1, s1], axis=1)
        elif (indx[i] and i==12):
            s1 = pd.DataFrame( {'SMA2': talib.SMA(np.array(df['Close'].values), timeperiod=60) }) 
            df1 = pd.concat([df1, s1], axis=1)

    #Delete 'A' column
    df1=df1.drop(columns=['A'])
    
    #Numpy array
    out=np.array(df1.values)
    out=out[~np.any(np.isnan(out), axis=1)]
    
    return out

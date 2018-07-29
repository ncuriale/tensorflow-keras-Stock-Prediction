import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, LSTM
from keras.preprocessing.sequence import TimeseriesGenerator

#Import scripts
from technicalData import *
from helpers import *
from buildAnalyze import *
###########################################################################################

#    Select Indicators to compute ----- 1-Use 0-Not Use
#     Open, High,   Low,    Close,  A.Close,    Vol,   
#     RSI,  ROC,    BETA,   STDDEV, WILLR,   SMA1,  
#     SMA2

indx=[0,   0,   0,   1,   0,   1,   \
      1,   1,   1,   1,   1,   1,   \
      1]

###########################################################################################

######MAIN######
def getStockData():
    
    #####HYPER-PARAMETERS
    layers = [32,256,128,64,32,1]     
    train_size=0.8
    future_window_size=10
    window_size = layers[0]
    epochs=10
    batch_size=128

    #Read-in and split stock data
    df = pd.read_csv('IAG.TO.csv')
    data_raw=np.array(df[['Close','Volume']].values)
    data_raw_w_Features=calcFeatures(df,indx)
    numFeatures=len(data_raw_w_Features[0])
    data_norm=normalize(data_raw_w_Features)
    index=np.array(df.index.values)

    # And now we can window the data using our windowing function
    X,y,indexes = series_to_seq(series = data_norm, indexes=index, \
                    window_size = window_size, future_window_size = future_window_size)
    
    # split our dataset into training / testing sets
    X_train, y_train, ind_train, X_test, y_test, ind_test = \
                    train_test_split(X, y, indexes, train_size)
    X_train, X_test = reshapeX(X_train,X_test,window_size,numFeatures)
    y_train, y_test, data_norm = y_train[:,0], y_test[:,0], data_norm[:,0] #Set output variables from features

    # Build an RNN to perform regression on our time series input/output data
    model = build_model(layers,batch_size,numFeatures)
    model.summary()

    # compile and fit the model
    model_fit=model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    plotHistory(model_fit)

    #Analyze predicitions
    train_predict, test_predict = predict_analyze(  \
                        X_train, y_train, X_test, y_test, model)

    ####simulate $$ over time
    init=10000
    simData=simulateTest(init,test_predict, y_test, data_raw[:,0], ind_test, future_window_size) 
    print(simData)
    print(simData[0]/init,len(test_predict))

    ### Plot
    plotPredictions(data_norm, train_predict, test_predict, X, \
                        train_size, window_size, future_window_size)

if __name__ == "__main__":
    getStockData()
    
    
    
    
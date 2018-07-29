import numpy as np
import matplotlib.pyplot as plt

def plotHistory(history):

    # summarize history for accuracy
    plt.plot(history.history['mean_squared_error'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plotPredictions(data, train_predict, test_predict, X, train_size, window_size, future_window_size):
    # plot original series
    plt.plot(data,color = 'k')
    
    # plot training set prediction
    train_len=trainLength(X,train_size)
    split_pt = train_len + window_size + future_window_size
    plt.plot(np.arange(window_size+future_window_size,split_pt,1),train_predict,color = 'b')
    plt.plot(np.arange(window_size,split_pt-future_window_size,1),train_predict,color = 'b',linestyle='--')
    
    # plot testing set prediction
    plt.plot(np.arange(split_pt,split_pt + len(test_predict),1),test_predict,color = 'r')
    plt.plot(np.arange(split_pt-future_window_size,split_pt + len(test_predict)-future_window_size,1),test_predict,color = 'r',linestyle='--')

    # pretty up graph
    plt.xlabel('day')
    plt.ylabel('(normalized) price of stock')
    plt.legend(['original series','training fit','testing fit'],loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.show()
    
def normalize(data):
    norm=np.zeros_like(data)
    for i in range(len(data[0])):
        col=data[:,i]
        dmin, dmax = np.min(col), np.max(col)
        dmid = (dmin+dmax)/2
        norm[:,i] = ( data[:,i]-dmid ) / (dmax-dmid)
        
    return norm

def unnormalize(data,data_raw):
    for i in range(len(data[0])):
        dmin, dmax = np.min(data_raw[:,i]), np.max(data_raw[:,i])
        dmid = (dmin+dmax)/2
        data[:,i] = data[:,i]*(dmax-dmid) + dmid
        
    return data

def train_test_split(X,y,indexes,train_size):
    train_len=trainLength(X,train_size)
    X_train, y_train, ind_train = X[:train_len,:], y[:train_len], indexes[:train_len]
    X_test, y_test, ind_test = X[train_len:,:], y[train_len:], indexes[train_len:]
    return X_train, y_train, ind_train, X_test, y_test, ind_test

def reshapeX(X_train,X_test,window_size,numFeatures):
    # Keras RNN LSTM module must be reshaped to [samples, window size, stepsize] 
    X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, numFeatures)))
    X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, numFeatures)))
        
    return X_train,X_test

### The function below transforms the input series and window-size into a set of input/output pairs for our RNN model
def series_to_seq(series,indexes,window_size,future_window_size):
        
    seq_length=len(series)-window_size-future_window_size        

    # containers for input/output pairs
    Xt = []
    yt = []
    indt = []

    for j in range(window_size, len(series)-future_window_size):
        Xt.append(series[j - window_size:j])
        yt.append(series[j + future_window_size])
        indt.append(indexes[j])
           
    X=np.asarray(Xt)
    y=np.asarray(yt)
    ind=np.asarray(indt)

    return X,y,ind

def trainLength(X,train_size):
    return int(len(X)*train_size)





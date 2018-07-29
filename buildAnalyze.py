from keras.models import Model
from keras.layers import Input, Dense, Activation, Dropout, LSTM
import numpy as np
    
def build_model(layers,batchSize,numFeatures):

    inputs = Input(shape=(layers[0], numFeatures))
    x = LSTM(layers[1], dropout=0.1, return_sequences=True)(inputs)
    x = LSTM(layers[2], dropout=0.1, return_sequences=True)(x)
    x = LSTM(layers[3], dropout=0.1)(x)    
    x = Dense(layers[4],activation="relu")(x)
    outputs = Dense(layers[5],activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss="mse", optimizer="rmsprop", metrics=['mean_squared_error'])

    return model

def predict_analyze(X_train, y_train, X_test, y_test, model):
    # generate predictions for training
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # print out training and testing errors
    training_error = model.evaluate(X_train, y_train, verbose=0)
    print('training error = ' + str(training_error))

    testing_error = model.evaluate(X_test, y_test, verbose=0)
    print('testing error = ' + str(testing_error))

    return train_predict, test_predict

def simulateTest(cashInit, prediction, actual, data, indexes, future_window_size):

    # Simulation data
    # $ in cash, $ invested, # of stocks, stock price bought]
    simData=[cashInit, 0, 0, None]

    #Go thru all data
    for i in range(len(prediction)):
        if simData[3]==None:
            if prediction[i]>actual[i]:
                simData[3] = data[indexes[i]]
                simData[2] = np.floor(simData[0]/simData[3])
                simData[1] = simData[0] - (simData[0]%simData[3])
                simData[0] -= simData[1]
        else:            
            if prediction[i]<actual[i]:
                simData[3] = None
                simData[0] += simData[2]*data[indexes[i]]
                simData[2] = 0
                simData[1] = 0

    return simData
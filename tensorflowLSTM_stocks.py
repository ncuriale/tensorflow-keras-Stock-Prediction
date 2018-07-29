import tensorflow as tf
import numpy as np

#Import scripts
from technicalData import *
from helpers import *

class Config(object):
    """
    Class to store parameters
    """

    def __init__(self, X_train, X_test):
        # Input data
        self.train_count = len(X_train)  # training series
        self.test_data_count = len(X_test)  # testing series
        self.n_steps = len(X_train[0])  # time_steps per series

        # Training
        self.learning_rate = 0.05
        self.lambda_loss_amount = 0.0015
        self.epochs = 100
        self.batch_size = 1500

        # LSTM structure
        self.n_inputs = len(X_train[0][0])  # Features count 
        self.n_hidden = 256  # Hidden neurons
        self.n_outputs = 1  # Final outputs
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_outputs]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_outputs]))
        }


def LSTM_Network(_X, config):
    """Function returns a TensorFlow RNN with two stacked LSTM cells
    Args:
        _X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
        config: Config for the neural network.
      return:
              : matrix  output shape [batch_size,n_classes]
    """

    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, config.n_inputs])
    # new shape: (n_steps*batch_size, n_input)

    # Linear activation
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, config.n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many to one" classifier
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, config.W['output']) + config.biases['output']

if __name__ == "__main__":

    ###########################################################################################

    #    Select Indicators to compute ----- 1-Use 0-Not Use
    #     Open, High,   Low,    Close,  A.Close,    Vol,   
    #     RSI,  ROC,    BETA,   STDDEV, WILLR,   SMA1,  
    #     SMA2

    indx=[0,   0,   0,   1,   0,   1,   \
          1,   1,   1,   1,   1,   1,   \
          1]

    ###########################################################################################
    
    ############## Prepare data ##############
    train_size=0.7
    future_window_size=10
    window_size = 30    
    
    #Read-in and split stock data
    df = pd.read_csv('SHOP.TO.csv')
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
    y_train, y_test, data_norm = y_train[:,0:1], y_test[:,0:1], data_norm[:,0:1] #Set output variables from features

    ############## Define config class ##############
    config = Config(X_train, X_test)
    print("Input shape, Output shape, Input mean, Input std dev")
    print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))

    ############## Build model ##############
    X = tf.placeholder(tf.float32, [None, config.n_steps, config.n_inputs])
    Y = tf.placeholder(tf.float32, [None, config.n_outputs])

    pred_Y = LSTM_Network(X, config)

    # Loss,optimizer,evaluation
    l2 = config.lambda_loss_amount * \
        sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    # Softmax loss and L2
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred_Y)) + l2
    optimizer = tf.train.AdamOptimizer(
        learning_rate=config.learning_rate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(pred_Y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

    ############## Train neural network ##############
    sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=False))
    init = tf.global_variables_initializer()
    sess.run(init)

    best_accuracy = 0.0
    # Start training for each batch and loop epochs
    for i in range(config.epochs):
        for start, end in zip(range(0, config.train_count, config.batch_size),
                              range(config.batch_size, config.train_count + 1, config.batch_size)):
            sess.run(optimizer, feed_dict={X: X_train[start:end],
                                           Y: y_train[start:end]})

        # Extract accuracy at each epoch
        pred_out, accuracy_out, loss_out = sess.run(
            [pred_Y, accuracy, cost], feed_dict={X: X_test, Y: y_test } )
        print("traing iter: {},".format(i) +
              " test accuracy : {},".format(accuracy_out) +
              " loss : {}".format(loss_out))
        best_accuracy = max(best_accuracy, accuracy_out)

    print("")
    print("final test accuracy: {}".format(accuracy_out))
    print("best epoch's test accuracy: {}".format(best_accuracy))
    print("")


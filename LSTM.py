import pandas as pd
from numpy import array, hstack
import tensorflow as tf
import torch
from keras.models import Sequential
# from tensorflow import keras
# from tensorflow.keras import layers
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
import matplotlib.pyplot as plt
import math
from tensorflow.keras.optimizers import SGD, Adam, Adamax, Adadelta, Adagrad


class LSTM_Model:
    def __init__(self, n_steps_in, n_steps_out):
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out

    def load_model(self, n_features):
        model = Sequential()
        model.add(LSTM(50, activation='relu', recurrent_activation='sigmoid', return_sequences=True,
                       input_shape=(self.n_steps_in, n_features)))
        model.add(LSTM(50, activation='relu', recurrent_activation='sigmoid', ))
        model.add(Dense(self.n_steps_out))
        # model.add(Dense(1)) ## one output

        optimizers = [tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, ),
                      tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07),
                      tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
                      tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07)]
        # tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.05, nesterov=False)]

        return model, optimizers

    def plot_tarin(self, optimizers, model, x_trin, x_test, x_val, y_trin, y_test, y_val):
        op = ['adam', 'adagrad', 'adamax', 'adadelta']  # , 'sgd']
        i = 0
        for optimizer in optimizers:
            model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mse'])
            # fit network
            history = model.fit(x_trin, y_trin, epochs=50, batch_size=1, validation_data=(x_test, y_test), verbose=2, shuffle=False)
            model.evaluate(x_val, y_val, batch_size=1, return_dict=True)
            # summarize history for loss
            plt.figure()
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.title(f'model loss {op[i]}')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend()
            i += 1

    def binary_acc(self, y_pred, y_test):
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        correct_results_sum = (y_pred_tag == y_test).sum().float()
        acc = correct_results_sum / y_test.shape[0]
        acc = torch.round(acc * 100)

        return acc

    def prep_data(self):
        X = pd.read_csv('WB_0.csv')
        X_temp = X.pop('Turbidity')
        X['Turbidity'] = X_temp
        X.pop('Measurement Date')
        # define input sequence
        in_seq1 = X['Water Temperature'].to_numpy()
        in_seq2 = X['Wave Height'].to_numpy()
        in_seq3 = X['Wave Period'].to_numpy()
        in_seq4 = X['Battery Life'].to_numpy()
        in_seq5 = X['Measurement Day'].to_numpy()
        in_seq6 = X['Measurement Month'].to_numpy()
        in_seq7 = X['Measurement Year'].to_numpy()
        in_seq8 = X['Measurement Hour'].to_numpy()
        in_seq9 = X['Latitude'].to_numpy()
        in_seq10 = X['class'].to_numpy()
        out_seq = X['Turbidity'].to_numpy()

        # convert to [rows, columns] structure
        in_seq1 = in_seq1.reshape((len(in_seq1), 1))
        in_seq2 = in_seq2.reshape((len(in_seq2), 1))
        in_seq3 = in_seq3.reshape((len(in_seq3), 1))
        in_seq4 = in_seq4.reshape((len(in_seq4), 1))
        in_seq5 = in_seq5.reshape((len(in_seq5), 1))
        in_seq6 = in_seq6.reshape((len(in_seq6), 1))
        in_seq7 = in_seq7.reshape((len(in_seq7), 1))
        in_seq8 = in_seq8.reshape((len(in_seq8), 1))
        in_seq9 = in_seq9.reshape((len(in_seq9), 1))
        in_seq10 = in_seq10.reshape((len(in_seq10), 1))
        out_seq = out_seq.reshape((len(out_seq), 1))
        # horizontally stack columns
        dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, in_seq6, in_seq7, in_seq8, in_seq9, in_seq10, out_seq))
        # covert into input/output
        x_trin, x_val, x_test, y_trin, y_val, y_test = self.split_sequences(dataset, self.n_steps_in, self.n_steps_out)
        # the dataset knows the number of features, e.g. 2
        n_features = x_trin.shape[2]
        print(x_trin.shape)
        print(x_val.shape)
        print(x_test.shape)

        return x_trin, x_val, x_test, y_trin, y_val, y_test, n_features

    # split a multivariate sequence into samples
    def split_sequences(self, sequences, n_steps_in, n_steps_out):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps_in
            out_end_ix = end_ix + n_steps_out - 1
            # check if we are beyond the dataset
            if out_end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix - 1:out_end_ix, -1]
            X.append(seq_x)
            y.append(seq_y)
        X = array(X)
        y = array(y)
        s = math.floor(X.shape[0] * 0.7)  # round down
        x_trin = X[:math.floor(s * 0.8)]
        y_trin = y[:math.floor(s * 0.8)]
        x_val = X[math.floor(s * 0.8):s]
        y_val = y[math.floor(s * 0.8):s]
        x_test = X[s:]
        y_test = y[s:]

        return x_trin, x_val, x_test, y_trin, y_val, y_test
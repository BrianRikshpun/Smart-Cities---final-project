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
from sklearn.preprocessing import MinMaxScaler
from pandas import Series
from tensorflow.keras.optimizers import SGD, Adam, Adamax, Adadelta, Adagrad


class LSTM_Model:
    def __init__(self, n_steps_in, n_steps_out):
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out

    def load_model(self, n_features):
        model = Sequential()
        model.add(LSTM(50, activation='relu', recurrent_activation='sigmoid', return_sequences=True, input_shape=(self.n_steps_in, n_features)))
        model.add(LSTM(50, activation='relu', recurrent_activation='sigmoid', ))
        model.add(Dense(self.n_steps_out))
        # model.add(Dense(1)) ## one output

        # model = Sequential()
        # model.add(LSTM(100, activation='relu', input_shape=(self.n_steps_in, n_features)))
        # model.add(RepeatVector(self.n_steps_out))
        # model.add(LSTM(100, activation='relu', return_sequences=True))
        # model.add(TimeDistributed(Dense(1)))

        optimizers = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        # tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07),
        # tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        # tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07)]
        # tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.05, nesterov=False)]

        return model, optimizers

    def plot_tarin(self, optimizers, model, x_trin, x_test, x_val, y_trin, y_test, y_val):
        op = ['adam']  #, 'adagrad', 'adamax', 'adadelta' , 'sgd']
        i = 0
        for optimizer in optimizers:
            model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy', 'mse'])
            # fit network
            history = model.fit(x_trin, y_trin, epochs=10, batch_size=7, validation_data=(x_test, y_test), verbose=0, shuffle=False)
            model.evaluate(x_val, y_val, batch_size=7, return_dict=True)
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
        in_seq4 = X['Battery Life'].to_numpy().astype(int)
        in_seq5 = X['Measurement Day'].to_numpy().astype(int)
        in_seq6 = X['Measurement Month'].to_numpy().astype(int)
        in_seq7 = X['Measurement Year'].to_numpy().astype(int)
        in_seq8 = X['Measurement Hour'].to_numpy().astype(int)
        in_seq9 = X['Latitude'].to_numpy()
        in_seq10 = X['class'].to_numpy().astype(int)
        out_seq = X['Turbidity'].to_numpy().astype(int)

        # # convert to [rows, columns] structure
        # in_seq1 = in_seq1.reshape((len(in_seq1), 1))
        # in_seq2 = in_seq2.reshape((len(in_seq2), 1))
        # in_seq3 = in_seq3.reshape((len(in_seq3), 1))
        # in_seq4 = in_seq4.reshape((len(in_seq4), 1))
        # in_seq5 = in_seq5.reshape((len(in_seq5), 1))
        # in_seq6 = in_seq6.reshape((len(in_seq6), 1))
        # in_seq7 = in_seq7.reshape((len(in_seq7), 1))
        # in_seq8 = in_seq8.reshape((len(in_seq8), 1))
        # in_seq9 = in_seq9.reshape((len(in_seq9), 1))
        # in_seq10 = in_seq10.reshape((len(in_seq10), 1))
        # out_seq = out_seq.reshape((len(out_seq), 1))
        # horizontally stack columns

        series_0 = Series(out_seq)
        values0 = series_0.values
        values0 = values0.reshape((len(values0), 1))
        scaler_0 = MinMaxScaler(feature_range=(0, 1))
        scaler_0 = scaler_0.fit(values0)
        out_seq = scaler_0.transform(values0)

        series_1 = Series(in_seq1)
        values1 = series_1.values
        values1 = values1.reshape((len(values1), 1))
        scaler1 = MinMaxScaler(feature_range=(0, 1))
        scaler1 = scaler1.fit(values1)
        in_seq1 = scaler1.transform(values1)

        series_2 = Series(in_seq2)
        values2 = series_2.values
        values2 = values2.reshape((len(values2), 1))
        scaler2 = MinMaxScaler(feature_range=(0, 1))
        scaler2 = scaler2.fit(values2)
        in_seq2 = scaler2.transform(values2)

        series_3 = Series(in_seq3)
        values3 = series_3.values
        values3 = values3.reshape((len(values3), 1))
        scaler3 = MinMaxScaler(feature_range=(0, 1))
        scaler3 = scaler3.fit(values3)
        in_seq3 = scaler3.transform(values3)

        series_4 = Series(in_seq4)
        values4 = series_4.values
        values4 = values4.reshape((len(values4), 1))
        scaler4 = MinMaxScaler(feature_range=(0, 1))
        scaler4 = scaler4.fit(values4)
        in_seq4 = scaler4.transform(values4)

        series5 = Series(in_seq5)
        values5 = series5.values
        values5 = values5.reshape((len(values5), 1))
        scaler5 = MinMaxScaler(feature_range=(0, 1))
        scaler5 = scaler5.fit(values5)
        in_seq5 = scaler5.transform(values5)

        series6 = Series(in_seq6)
        values6 = series6.values
        values6 = values6.reshape((len(values6), 1))
        scaler6 = MinMaxScaler(feature_range=(0, 1))
        scaler6 = scaler6.fit(values6)
        in_seq6 = scaler6.transform(values6)

        series7 = Series(in_seq7)
        values7 = series7.values
        values7 = values7.reshape((len(values7), 1))
        scaler7 = MinMaxScaler(feature_range=(0, 1))
        scaler7 = scaler7.fit(values7)
        in_seq7 = scaler7.transform(values7)

        series8 = Series(in_seq8)
        values8 = series8.values
        values8 = values8.reshape((len(values8), 1))
        scaler8 = MinMaxScaler(feature_range=(0, 1))
        scaler8 = scaler8.fit(values8)
        in_seq8 = scaler8.transform(values8)

        series9 = Series(in_seq9)
        values9 = series9.values
        values9 = values9.reshape((len(values9), 1))
        scaler9 = MinMaxScaler(feature_range=(-1, 0))
        scaler9 = scaler9.fit(values9)
        in_seq9 = scaler9.transform(values9)

        series10 = Series(in_seq10)
        values10 = series10.values
        values10 = values10.reshape((len(values10), 1))
        scaler10 = MinMaxScaler(feature_range=(0, 1))
        scaler10 = scaler10.fit(values10)
        in_seq10 = scaler10.transform(values10)

        dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq5, in_seq6, in_seq7, in_seq8, in_seq9, in_seq10, out_seq))
        # covert into input/output
        x_trin, x_val, x_test, y_trin, y_val, y_test, X_tv, y_tv = self.split_sequences(dataset, self.n_steps_in, self.n_steps_out)
        # the dataset knows the number of features, e.g. 2
        n_features = x_trin.shape[2]


        return x_trin, x_val, x_test, y_trin, y_val, y_test, X_tv, y_tv, n_features, scaler_0

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

        s = math.floor(X.shape[0] * 0.85)  # round down
        x_trin = X[:math.floor(s * 0.85)]
        y_trin = y[:math.floor(s * 0.85)]
        x_val = X[math.floor(s * 0.85):s]
        y_val = y[math.floor(s * 0.85):s]
        x_test = X[s:]
        y_test = y[s:]
        X_tv = X[:s]
        y_tv = y[:s]

        return x_trin, x_val, x_test, y_trin, y_val, y_test, X_tv, y_tv
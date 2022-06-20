import math
import torch
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import Series
from numpy import array, hstack
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout , RepeatVector, TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam #, Adamax, Adadelta, Adagrad, SGD
# from tensorflow import keras
# from tensorflow.keras import layers


class LSTM_M:
    def __init__(self, n_steps_in, n_steps_out):
        self.n_steps_in = n_steps_in
        self.n_steps_out = n_steps_out

    def prep_data_WB(self):
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

        # the dataset knows the number of features
        n_features = x_trin.shape[2]

        return x_trin, x_val, x_test, y_trin, y_val, y_test, X_tv, y_tv, n_features, scaler_0

    def prep_data_WWB(self):
        X = pd.read_csv('wWB_0.csv')
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
        in_seq11 = X['Air Temperature'].to_numpy()
        in_seq12 = X['Wet Bulb Temperature'].to_numpy()
        in_seq13 = X['Humidity'].to_numpy()
        in_seq14 = X['Total Rain'].to_numpy()
        in_seq15 = X['Wind Direction'].to_numpy()
        in_seq16 = X['Wind Speed'].to_numpy()
        in_seq17 = X['Maximum Wind Speed'].to_numpy()
        in_seq18 = X['Barometric Pressure'].to_numpy()
        in_seq19 = X['Solar Radiation'].to_numpy()
        in_seq20 = X['Heading'].to_numpy()
        out_seq = X['Turbidity'].to_numpy()


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

        series11 = Series(in_seq11)
        values11 = series11.values
        values11 = values11.reshape((len(values11), 1))
        scaler11 = MinMaxScaler(feature_range=(0, 1))
        scaler11 = scaler11.fit(values11)
        in_seq11 = scaler11.transform(values11)

        series12 = Series(in_seq12)
        values12 = series12.values
        values12 = values12.reshape((len(values12), 1))
        scaler12 = MinMaxScaler(feature_range=(0, 1))
        scaler12 = scaler12.fit(values12)
        in_seq12 = scaler12.transform(values12)

        series13 = Series(in_seq13)
        values13 = series13.values
        values13 = values13.reshape((len(values13), 1))
        scaler13 = MinMaxScaler(feature_range=(0, 1))
        scaler13 = scaler13.fit(values13)
        in_seq13 = scaler13.transform(values13)

        series14 = Series(in_seq14)
        values14 = series14.values
        values14 = values14.reshape((len(values14), 1))
        scaler14 = MinMaxScaler(feature_range=(0, 1))
        scaler14 = scaler14.fit(values14)
        in_seq14 = scaler14.transform(values14)

        series15 = Series(in_seq15)
        values15 = series15.values
        values15 = values15.reshape((len(values15), 1))
        scaler15 = MinMaxScaler(feature_range=(0, 1))
        scaler15 = scaler15.fit(values15)
        in_seq15 = scaler15.transform(values15)

        series16 = Series(in_seq16)
        values16 = series16.values
        values16 = values16.reshape((len(values16), 1))
        scaler16 = MinMaxScaler(feature_range=(0, 1))
        scaler16 = scaler16.fit(values16)
        in_seq16 = scaler16.transform(values16)

        series17 = Series(in_seq17)
        values17 = series17.values
        values17 = values17.reshape((len(values17), 1))
        scaler17 = MinMaxScaler(feature_range=(0, 1))
        scaler17 = scaler17.fit(values17)
        in_seq17 = scaler17.transform(values17)

        series18 = Series(in_seq18)
        values18 = series18.values
        values18 = values18.reshape((len(values18), 1))
        scaler18 = MinMaxScaler(feature_range=(0, 1))
        scaler18 = scaler18.fit(values18)
        in_seq18 = scaler18.transform(values18)

        series19 = Series(in_seq19)
        values19 = series19.values
        values19 = values19.reshape((len(values19), 1))
        scaler19 = MinMaxScaler(feature_range=(0, 1))
        scaler19 = scaler19.fit(values19)
        in_seq19 = scaler19.transform(values19)

        series20 = Series(in_seq20)
        values20 = series20.values
        values20 = values20.reshape((len(values20), 1))
        scaler20 = MinMaxScaler(feature_range=(0, 1))
        scaler20 = scaler20.fit(values20)
        in_seq20 = scaler20.transform(values20)

        dataset = hstack((in_seq1, in_seq2, in_seq3, in_seq4, in_seq10, in_seq5, in_seq6, in_seq7, in_seq8, in_seq9,  in_seq17,
                          in_seq11, in_seq12, in_seq13, in_seq14, in_seq15, in_seq16, in_seq18, in_seq19, in_seq20, out_seq))
        # covert into input/output
        x_trin, x_val, x_test, y_trin, y_val, y_test, X_tv, y_tv = self.split_sequences(dataset, self.n_steps_in, self.n_steps_out)

        # the dataset knows the number of features
        n_features = x_trin.shape[2]

        return x_trin, x_val, x_test, y_trin, y_val, y_test, X_tv, y_tv, n_features, scaler_0

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

        s = math.floor(X.shape[0] * 0.8)  # round down
        x_trin = X[:math.floor(s * 0.85)]
        y_trin = y[:math.floor(s * 0.85)]
        x_val = X[math.floor(s * 0.85):s]
        y_val = y[math.floor(s * 0.85):s]
        x_test = X[s:]
        y_test = y[s:]
        X_tv = X[:s]
        y_tv = y[:s]

        return x_trin, x_val, x_test, y_trin, y_val, y_test, X_tv, y_tv

    def load_model(self, n_features):
        print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;')
        print(n_features)
        # model = Sequential()
        # model.add(LSTM(50, activation='relu', return_sequences=True, nput_shape=(self.n_steps_in, n_features, 1)))
        # model.add(LSTM(50, activation='relu'))
        # model.add(Dense(self.n_steps_out))

        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(self.n_steps_in, n_features)))
        model.add(RepeatVector(self.n_steps_out))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))

        optimizers = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        # tf.keras.optimizers.Adagrad(learning_rate=0.001, initial_accumulator_value=0.1, epsilon=1e-07),
        # tf.keras.optimizers.Adamax(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
        # tf.keras.optimizers.Adadelta(learning_rate=0.001, rho=0.95, epsilon=1e-07)]
        # tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.05, nesterov=False)]

        model.compile(optimizer=optimizers, loss='mean_squared_error', metrics=['accuracy', 'mse'])

        return model

    def plot_tarin(self, model, x_trin, x_test, x_val, y_trin, y_test, y_val, epochs):
        op = ['adam']  #, 'adagrad', 'adamax', 'adadelta' , 'sgd']

        # fit network
        history = model.fit(x_trin, y_trin, epochs=epochs, batch_size=1,
                            validation_data=(x_test, y_test), verbose=0, shuffle=False)

        model.evaluate(x_val, y_val, batch_size=1, return_dict=True)
        # summarize history for loss
        plt.figure()
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title(f'model loss {op}')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()

        return model


class LSTM_model:

    def model(self, X_train, number_units, dropout_fraction):
        model = Sequential()
        # Layer 1
        model.add(LSTM(
            units=number_units,
            return_sequences=True,
            input_shape=(X_train.shape[1], 1))
        )
        model.add(Dropout(dropout_fraction))
        # Layer 2
        model.add(LSTM(units=number_units, return_sequences=True))
        model.add(Dropout(dropout_fraction))
        # Layer 3
        model.add(LSTM(units=number_units))
        model.add(Dropout(dropout_fraction))
        # Output layer
        model.add(Dense(1))
        # Compile the model
        optimize = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

        model.compile(optimizer=optimize, loss="mean_squared_error", metrics=['acc'])
        return model

    # This function accepts the column number for the features (X) and the target (y)
    # It chunks the data up with a rolling window of Xt-n to predict Xt
    # It returns a numpy array of X any y
    def window_data(self, df, window, feature_col_number, target_col_number):
        X = []
        y = []
        for i in range(len(df) - window - 1):
            features = df.iloc[i:(i + window), 0:feature_col_number]
            target = df.iloc[(i + window), target_col_number]
            X.append(features)
            y.append(target)
        return np.array(X), np.array(y).reshape(-1, 1)

    def scaler_func(self, X, y, X_train, X_test, y_train, y_test):
        # Use the MinMaxScaler to scale data between 0 and 1.
        scaler = MinMaxScaler()
        scaler.fit(X.reshape(X.shape[0], X.shape[1] * X.shape[2]))
        X_train = scaler.transform(X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        X_test = scaler.transform(X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
        scaler.fit(y)
        y_train = scaler.transform(y_train)
        y_test = scaler.transform(y_test)
        return X_train, X_test, y_train, y_test, scaler

    def prep_df(self, csv_path):
        # Load the fear and greed sentiment data for Bitcoin
        df1 = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['Water Temperature'].sort_index()
        df2 = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['Latitude'].sort_index()
        df3 = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['Wave Height'].sort_index()
        df4 = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['Wave Period'].sort_index()
        df5 = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['Battery Life'].sort_index().astype(int)
        # df6 = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['Measurement Day'].sort_index()
        # df8 = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['Measurement Month'].sort_index()
        # df9 = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['Measurement Year'].sort_index()
        df10 = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['Measurement Hour'].sort_index().astype(int)
        df11 = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['class'].sort_index().astype(int)
        df_t = pd.read_csv(csv_path, index_col="Measurement Date", infer_datetime_format=True, parse_dates=True)['Turbidity'].sort_index().astype(int)

        # Join the data into a single DataFrame
        df = df1.to_frame().join(df2, how="inner")
        df = df.join(df3, how='inner')
        df = df.join(df4, how='inner')
        # df = df.join(df4, how='inner')
        df = df.join(df5, how='inner')
        # df = df.join(df6, how='inner')
        # df = df.join(df8, how='inner')
        # df = df.join(df9, how='inner')
        df = df.join(df10, how='inner')
        df = df.join(df11, how='inner')
        df = df.join(df_t, how='inner')
        return df
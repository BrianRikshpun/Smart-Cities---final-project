import pandas as pd
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
from Preprocess import Preprocess
from ClassicModels import ClassicModels
from Visualization import Visualization
from Data_Loader import prep_data_loder
from nn_b_c import set_model
from LSTM import LSTM_model, LSTM_M
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
import math
from sklearn.metrics import confusion_matrix, classification_report


def start_lstm_2():
    # Predict Closing Prices using a 10 day window of previous closing prices
    # Then, experiment with window sizes anywhere from 1 to 10 and see how the model performance changes
    window_size = 19

    feature_column = 6
    target_column = 7
    number_units = 50
    dropout_fraction = 0.3
    epochs = 10

    lstm = LSTM_model()
    df = lstm.prep_df(csv_path="WB_0.csv")

    X, y = lstm.window_data(df, window_size, feature_column, target_column)
    # Use 70% of the data for training and the remaineder for testing
    split = int(0.7 * len(X))
    X_train = X[: split]
    X_test = X[split:]
    y_train = y[: split]
    y_test = y[split:]

    X_train, X_test, y_train, y_test, scaler = lstm.scaler_func(X, y, X_train, X_test, y_train, y_test)
    # Reshape the features for the model
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = lstm.model(X_train, number_units, dropout_fraction)
    history = model.fit(X_train, y_train, epochs=epochs, shuffle=False, batch_size=1, validation_data=(X_test, y_test), verbose=1)
    # Evaluate the model
    model.evaluate(X_test, y_test)

    # Make some predictions
    predicted = model.predict(X_test)
    # Recover the original prices instead of the scaled version
    predicted_Turbidity = scaler.inverse_transform(predicted)
    real_Turbidity = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.subplot()
    # Create a DataFrame of Real and Predicted values
    Turbidity = pd.DataFrame({
        "Real": real_Turbidity.ravel(),
        "Predicted": predicted_Turbidity.ravel()},
        index=df.index[-len(real_Turbidity):])
    Turbidity.plot(title="Turbidity Real vs. Predicted", figsize=(10, 5))
    plt.show()

    plt.subplot()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('model loss adam')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()

    plt.show()


def start_lstm():

    n_steps_in_r = 30
    n_steps_out_r = 14
    epochs = 12
    Precision = []

    df = pd.read_csv('WB_0.csv')
    ture_bc_test = df['class'].astype(int)
    split = math.floor(df.shape[0] * 0.85)
    ture_bc_test = ture_bc_test[split:]
    df_LSTM_ROC = pd.DataFrame()

    # ttt1 = 0
    # ttt0 = 0
    # for c in ture_bc_test:
    #     # print(c)
    #     if c == 1:
    #         ttt1 += 1
    #     else:
    #         ttt0 += 1
    #
    # print(ttt1, ttt0, len(ture_bc_test))

    for n_steps_out in range(2, n_steps_out_r+1):
        for n_steps_in in range(2, n_steps_in_r+1):
            if n_steps_in >= n_steps_out:
                lstm = LSTM_M(n_steps_in=n_steps_in, n_steps_out=n_steps_out)
                x_trin, x_val, x_test, y_trin, y_val, y_test, X_tv, y_tv, n_features, scaler_0 = lstm.prep_data()
                model = lstm.load_model(n_features=n_features)

                # fit network and plot loss
                model = lstm.plot_tarin(model, x_trin, x_test, x_val, y_trin, y_test, y_val, epochs)

                # evaluate network
                model.evaluate(x_val, y_val, batch_size=1)
                yhat = model.predict(x_test, verbose=0)

                pre = []
                for p in yhat:
                    pp = []
                    for i in p:
                        pp.append(scaler_0.inverse_transform(i.reshape(1, -1)).tolist()[0])
                    pre.append(pp)

                tp = 0
                tn = 0
                fp = 0
                fn = 0
                i = 0
                for p in pre:
                    if i < (len(ture_bc_test) - n_steps_out):
                        if p[n_steps_out-1][0] >= 5 and ture_bc_test.iloc[i+n_steps_out] == 1:
                            tp += 1
                        if p[n_steps_out-1][0] >= 5 and ture_bc_test.iloc[i+n_steps_out] == 0:
                            fn += 1
                        if p[n_steps_out-1][0] < 5 and ture_bc_test.iloc[i+n_steps_out] == 1:
                            fp += 1
                        if p[n_steps_out-1][0] < 5 and ture_bc_test.iloc[i+n_steps_out] == 0:
                            tn += 1
                    i+=1

                # print('------------------------')
                if tp == 0 and fn == 0 or tp == 0 and fp == 0:
                    print("tp-11, fp-10, tn-00, fn-01")
                    print(" ", tp, "   ", fp, "   ", tn, "    ", fn)
                    Sensitivity = -1.0
                    precisions = -1.0
                else:
                    print("tp-11, fp-10, tn-00, fn-01")
                    print(" ", tp, "   ", fp, "   ",  tn, "    ",  fn)

                    Sensitivity = tp / (tp + fn)
                    precisions = tp / (tp + fp)
                Precision.append([n_steps_in, n_steps_out-1, Sensitivity, precisions, tp, fp, tn, fn])
                df_LSTM_ROC = df_LSTM_ROC.append({'in': n_steps_in, 'out': n_steps_out-1, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}, ignore_index=True)
            else:
                pass
    df_LSTM_ROC.to_csv('wb_lstm.csv')

    # plt.show()


def stat_nn():

    WB = pd.read_csv("WB_0.csv")  # Water data
    WWB = pd.read_csv('wWB_0.csv')  # Water and weather data

    # hyper-parameters - 2 layers nn
    EPOCHS = 500
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001

    dl = prep_data_loder()
    train_loader, test_loader, X_test, y_test = dl.data_loder(WB, BATCH_SIZE)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sm = set_model()
    model, criterion, optimizer = set_model.load_model(LEARNING_RATE, device)

    model.train()
    for e in range(1, EPOCHS + 1):
        epoch_loss = 0
        epoch_acc = 0
        for X_batch, y_batch in train_loader:

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            xbs = X_batch.size(dim=0)

            if xbs == BATCH_SIZE:
                y_pred = model(X_batch)

                loss = criterion(y_pred, y_batch.unsqueeze(1))
                acc = sm.binary_acc(y_pred, y_batch.unsqueeze(1))

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        if e % 100 == 0 or e == 1:
            print(f'Epoch {e + 0:03}: | Loss: {epoch_loss / len(train_loader):.5f} |'
                  f' Acc: {epoch_acc / len(train_loader):.3f}')

    y_pred_list = []
    model.eval()
    # with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().detach().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    confusion_matrix(y_test, y_pred_list)
    print(classification_report(y_test, y_pred_list))


def startML():

    WB = pd.read_csv("WB_0.csv") #Water data
    WWB = pd.read_csv('wWB_0.csv') #Water and weather data
    fig_size = (10, 8)

    ShiftsWB = []
    ShiftsWWB = []

    #define the models
    models = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]

    #define the search space
    space = [[{"penalty": ['none', 'l2'], "C": [0.5, 1.0, 5.0], "solver": ["newton-cg", "lbfgs", "sag"]}],
             [{"n_neighbors": [2, 3, 5], "algorithm": ["ball_tree", "kd_tree", "brute"], "p" : [1,2]}],
             [{"criterion": ['gini', 'entropy', 'log_loss'], "max_depth": [3, 5 , 7], "min_samples_leaf" : [1,2,3] , "max_features" : ['auto', 'sqrt' , 'log2']}],
             [{"max_depth": [3, 5],"max_depth": [3, 5 , 7], "criterion": ['gini', 'entropy'], "n_estimators": [50, 70, 100]}]]


    for i in range(-7,0): #shift the class between -7 to -1
        ShiftsWB.append(makeShifts(WB, i))
        ShiftsWWB.append(makeShifts(WWB, i))


    for i in range(7):
        shift = (i + 1)*-1
        pre_WB = Preprocess()
        X_train1_WB, X_test_WB, y_train1_WB, y_test_WB = pre_WB.SplitPcaScale(ShiftsWB[i], 'WB shift = ' + str(shift) , fisi=fig_size)
        X_train_WB, y_train_WB = pre_WB.smote(X_train1_WB, y_train1_WB)
        pre_WB.plot_2d_space(X_train_WB, y_train_WB, X_train1_WB, y_train1_WB, label='SMOTE', fisi=fig_size)

        ModelsF = ClassicModels(models, space, X_train_WB, X_test_WB, y_train_WB, y_test_WB)
        res_data = ModelsF.FindBestParams(models, space, X_train_WB, X_test_WB, y_train_WB, y_test_WB)

        Visualizations1 = Visualization(res_data, fig_size)
        Visualizations1.ShowAUC('WB shift = ' + str(shift))
        Visualizations1.Show_Confussion_Matrix(X_test_WB, y_test_WB)
        Visualizations1.ShowRoc('WB shift = ' + str(shift))

        #----------WWB------------
        pre_WWB = Preprocess()
        X_train1_WWB, X_test_WWB, y_train1_WWB, y_test_WWB = pre_WWB.SplitPcaScale(ShiftsWWB[i], 'WWB shift = ' + str(shift), fisi=fig_size)
        X_train_WWB, y_train_WWB = pre_WWB.smote(X_train1_WWB, y_train1_WWB)
        pre_WWB.plot_2d_space(X_train_WWB, y_train_WWB, X_train1_WWB, y_train1_WWB, label='SMOTE', fisi=fig_size)

        ModelsF = ClassicModels(models, space, X_train_WWB, X_test_WWB, y_train_WWB, y_test_WWB)
        res_data = ModelsF.FindBestParams(models, space, X_train_WWB, X_test_WWB, y_train_WWB, y_test_WWB)

        Visualizations2 = Visualization(res_data, fig_size)
        Visualizations2.ShowAUC('WWB shift = ' + str(shift))
        Visualizations2.Show_Confussion_Matrix(X_test_WWB, y_test_WWB)
        Visualizations2.ShowRoc('WWB shift = ' + str(shift))
        plt.show()


def makeShifts(data, n):
    t = data
    t['class'] = t['class'].shift(n).dropna().astype(int)
    t = t.dropna()
    t['class'] = t['class'].astype(int)

    return t


if __name__ == '__main__':
    # startML()
    # stat_nn()
    start_lstm()
    # start_lstm_2()

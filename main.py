import pandas as pd
import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt
from Preprocess import Preprocess
from ClassicModels import ClassicModels
from Visualization import Visualization
from Data_Loader import prep_data_loder
from nn_b_c import set_model
from LSTM import LSTM_Model
from sklearn.linear_model import LogisticRegression
# from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
import math
from sklearn.metrics import confusion_matrix, classification_report


def start_lstm():

    n_steps_in_r = 8
    n_steps_out_r = 5
    epochs = 10
    hh = []
    prec = []
    df = pd.read_csv('WB_0.csv')
    ture_t = df['class'].astype(int)
    s = math.floor(df.shape[0] * 0.85)
    ture_t = ture_t[s:]
    ddf = pd.DataFrame()


    ttt1 = 0
    ttt0 = 0
    for c in ture_t:
        # print(c)
        if c == 1:
            ttt1 += 1
        else:
            ttt0 += 1

    print(ttt1, ttt0, len(ture_t))

    for n_steps_out in range(2, n_steps_out_r+1):
        for n_steps_in in range(2, n_steps_in_r+1):
            if n_steps_in >= n_steps_out:
                lstm = LSTM_Model(n_steps_in=n_steps_in, n_steps_out=n_steps_out)
                x_trin, x_val, x_test, y_trin, y_val, y_test, X_tv, y_tv, n_features, scaler_0 = lstm.prep_data()
                model, optimizers = lstm.load_model(n_features=n_features)
                model.compile(optimizer=optimizers, loss='mean_squared_error', metrics=['accuracy', 'mse'])

                # fit network

                history = model.fit(x_trin, y_trin, epochs=epochs, batch_size=1, validation_data=(x_test, y_test), verbose=0, shuffle=False)
                # history.appenmodel.evaluate(x_val, y_val, batch_size=1, return_dict=True)

                # evaluate network
                hh.append(history)

                # plt.figure()
                plt.plot(history.history['loss'], label=f'train {n_steps_in} {n_steps_out-1}')
                plt.plot(history.history['val_loss'], label=f'test {n_steps_in} {n_steps_out-1}')
                plt.title(f'model loss adam learn_in: {n_steps_in} ||| predict_out: {n_steps_out-1}')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend()
                plt.tight_layout()
                # y_pred = model(x_test)

                yhat = model.predict(x_test, verbose=0)
                # yhat = model.predict_on_batch(x_test)
                # print(yhat)
                pre = []
                for p in yhat:
                    pp = []
                    # print(p)
                    for i in p:
                        pp.append(scaler_0.inverse_transform(i.reshape(1, -1)).tolist()[0])
                    pre.append(pp)

                tp = 0
                tn = 0
                fp = 0
                fn = 0
                i = 0
                for p in pre:
                    cc = 0
                    if i < (len(ture_t) - n_steps_out):
                        if p[n_steps_out-1][0]*2 >= 5 and ture_t.iloc[i+n_steps_out] == 1:
                            tp += 1
                        if p[n_steps_out-1][0]*2 >= 5 and ture_t.iloc[i+n_steps_out] == 0:
                            fn += 1
                        if p[n_steps_out-1][0]*2 < 5 and ture_t.iloc[i+n_steps_out] == 1:
                            fp += 1
                        if p[n_steps_out-1][0]*2 < 5 and ture_t.iloc[i+n_steps_out] == 0:
                            tn += 1
                    #
                    # for pp in p:
                    #     # print(pp)
                    #     if i < (len(ture_t) - n_steps_out):
                    #         if pp[0] >= 5 and ture_t[i: i+n_steps_out].to_numpy()[cc] == 1:
                    #             tp += 1
                    #             # print("class 1")
                    #         if pp[0] >= 5 and ture_t[i: i+n_steps_out].to_numpy()[cc] == 0:
                    #             fn += 1
                    #         if pp[0] < 5 and ture_t[i: i+n_steps_out].to_numpy()[cc] == 1:
                    #             fp += 1
                    #         if pp[0] < 5 and ture_t[i: i+n_steps_out].to_numpy()[cc] == 0:
                    #             tn += 1
                    #     cc+=1
                    i+=1

                # print('------------------------')
                if tp == 0 and fn == 0 or tp == 0 and fp == 0:
                    print("tp-11, fp-10, tn-00, fn-01")
                    print(" ", tp, "   ", fp, "   ", tn, "    ", fn)
                    senseti = -1.0
                    preci = -1.0
                else:
                    print("tp-11, fp-10, tn-00, fn-01")
                    print(" ", tp, "   ", fp, "   ",  tn, "    ",  fn)
                    senseti = tp / (tp + fn)
                    preci = tp / (tp + fp)
                prec.append([n_steps_in, n_steps_out-1, senseti, preci, tp, fp, tn, fn])
                ddf = ddf.append({'in': n_steps_in, 'out': n_steps_out-1, 'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}, ignore_index=True)
            else:
                pass
    ddf.to_csv('wb_lstm.csv')
    plt.figure()
    for i in range(len(prec)):
        plt.bar(i+1, prec[i][2], label=f'senseti {prec[i][0]} -> {prec[i][1]}')
        plt.bar(i+1, prec[i][3], label=f'preci {prec[i][0]} -> {prec[i][1]}')
        plt.ylabel('')
        plt.xlabel('epx')
        plt.legend()
        plt.tight_layout()
        print(f'{prec[i][0]} -> {prec[i][1]}')
        print('senseti -- prec -- tp fp tn fn')
        print(prec[i][:])
    plt.show()


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

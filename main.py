import pandas as pd
from matplotlib import pyplot as plt
from Preprocess import Preprocess
from ClassicModels import ClassicModels
from Visualization import Visualization
from Data_Loader import prep_data_loder
from nn_b_c import set_model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import torch
from sklearn.metrics import confusion_matrix, classification_report


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
    model, criterion, optimizer = sm.load_model(LEARNING_RATE, device)

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

    #define the models
    models = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]

    #define the search space
    space = [[{"penalty": ['none', 'l2'], "C": [0.5, 1.0, 5.0], "solver": ["newton-cg", "lbfgs", "sag"]}],
             [{"n_neighbors": [2, 3, 5], "algorithm": ["ball_tree", "kd_tree", "brute"], "p" : [1,2]}],
             [{"criterion": ['gini', 'entropy', 'log_loss'], "max_depth": [3, 5 , 7], "min_samples_leaf" : [1,2,3] , "max_features" : ['auto', 'sqrt' , 'log2']}],
             [{"max_depth": [3, 5],"max_depth": [3, 5 , 7], "criterion": ['gini', 'entropy'], "n_estimators": [50, 70, 100]}]]


    pre_WB = Preprocess()
    X_train1_WB, X_test_WB, y_train1_WB, y_test_WB = pre_WB.SplitPcaScale(WB, 'WB', fisi=fig_size)
    X_train_WB, y_train_WB = pre_WB.smote(X_train1_WB, y_train1_WB)
    pre_WB.plot_2d_space(X_train_WB, y_train_WB, X_train1_WB, y_train1_WB, label='SMOTE', fisi=fig_size)

    ModelsF = ClassicModels(models, space, X_train_WB, X_test_WB, y_train_WB, y_test_WB)
    res_data = ModelsF.FindBestParams(models, space, X_train_WB, X_test_WB, y_train_WB, y_test_WB)

    Visualizations1 = Visualization(res_data, fig_size)
    Visualizations1.ShowAUC("WB")
    Visualizations1.Show_Confussion_Matrix(X_test_WB, y_test_WB)
    Visualizations1.ShowRoc("WB")

    pre_WWB = Preprocess()
    X_train1_WWB, X_test_WWB, y_train1_WWB, y_test_WWB = pre_WWB.SplitPcaScale(WWB, 'WWB', fisi=fig_size)
    X_train_WWB, y_train_WWB = pre_WWB.smote(X_train1_WWB, y_train1_WWB)
    pre_WWB.plot_2d_space(X_train_WWB, y_train_WWB, X_train1_WWB, y_train1_WWB, label='SMOTE', fisi=fig_size)

    ModelsF = ClassicModels(models, space, X_train_WWB, X_test_WWB, y_train_WWB, y_test_WWB)
    res_data = ModelsF.FindBestParams(models, space, X_train_WWB, X_test_WWB, y_train_WWB, y_test_WWB)

    Visualizations2 = Visualization(res_data, fig_size)
    Visualizations2.ShowAUC("WWB")
    Visualizations2.Show_Confussion_Matrix(X_test_WWB, y_test_WWB)
    Visualizations2.ShowRoc("WWB")
    plt.show()


if __name__ == '__main__':
    # startML()

    stat_nn()


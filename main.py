import pandas as pd
from matplotlib import pyplot as plt
from Preprocess import Preprocess
from ClassicModels import ClassicModels
from Visualization import Visualization
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def startML():

    WB = pd.read_csv("WB_0.csv") #Water data

    WWB = pd.read_csv('wWB_0.csv') #Water and weather data
    fig_size = (8, 15)

    #print(WB['class'].value_counts())
    # print(X_train_WB)
    #
    # print("y_train_WB")
    # print(y_train_WB)

    #define the models
    models = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier()]

    #define the search space
    space = [[{"penalty": ['none', 'l2'], "C": [0.5, 1.0, 5.0], "solver": ["newton-cg", "lbfgs", "sag"]}],
             [{"n_neighbors": [2, 3, 5], "algorithm": ["ball_tree", "kd_tree", "brute"], "p" : [1,2]}],
             [{"criterion": ['gini', 'entropy', 'log_loss'], "max_depth": [3, 5 , 7], "min_samples_leaf" : [1,2,3] , "max_features" : ['auto', 'sqrt' , 'log2']}],
             [{"max_depth": [3, 5],"max_depth": [3, 5 , 7], "criterion": ['gini', 'entropy'], "n_estimators": [50, 70, 100]}]]


    pre_WB = Preprocess()

    X_train_WB, X_test_WB, y_train_WB, y_test_WB = pre_WB.SplitPcaScale(WB, 'WB', fisi=fig_size)
    X_train_WB, y_train_WB = pre_WB.smote(X_train_WB, y_train_WB)

    ModelsF = ClassicModels(models, space, X_train_WB, X_test_WB, y_train_WB, y_test_WB)
    res_data = ModelsF.FindBestParams(models, space, X_train_WB, X_test_WB, y_train_WB, y_test_WB)

    Visualizations1 = Visualization(res_data, fig_size)
    Visualizations1.ShowAUC("WB")
    Visualizations1.Show_Confussion_Matrix(X_test_WB, y_test_WB)
    Visualizations1.ShowRoc("WB")

    pre_WWB = Preprocess()
    X_train_WWB, X_test_WWB, y_train_WWB, y_test_WWB = pre_WWB.SplitPcaScale(WWB, 'WWB', fisi=fig_size)

    ModelsF = ClassicModels(models, space, X_train_WWB, X_test_WWB, y_train_WWB, y_test_WWB)
    res_data = ModelsF.FindBestParams(models, space, X_train_WWB, X_test_WWB, y_train_WWB, y_test_WWB)

    Visualizations2 = Visualization(res_data, fig_size)
    Visualizations2.ShowAUC("WWB")
    Visualizations2.Show_Confussion_Matrix(X_test_WWB, y_test_WWB)
    Visualizations2.ShowRoc("WWB")
    plt.show()

    # Visualizations = Visualization(res_data)
    # Visualizations.ShowAUC(res_data)
    # Visualizations.ShowConfussionMatrix(res_data)
    # Visualizations.ShowRoc(res_data)

if __name__ == '__main__':
    startML()

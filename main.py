import pandas as pd
from Preprocess import Preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def startML():
    WB = pd.read_csv("WB.csv") #Water data
    WWB = pd.read_csv('WWB.csv') #Water and weather data

    #define the models
    models = [LogisticRegression(), GaussianNB(), KNeighborsClassifier() ]

    #define the search space
    space = [[{"penalty": ['none', 'l2'], "C": [0.5, 1.0, 5.0], "solver": ["newton-cg", "lbfgs", "sag"]}],
             [],
             [{"n_neighbors": [2, 3, 5], "algorithm": ["ball_tree", "kd_tree", "brute"], "p" : [1,2]}],
             [{"n_estimators": [50, 70, 100], "max_depth": [3, 5]}],
             [{"max_depth": [3, 5], "learning_rate": [0.1, 0.3], "n_estimators": [50, 70, 100]}]]




    pre_WB = Preprocess(WB)
    X_train_WB, X_test_WB, y_train_WB, y_test_WB = pre_WB.SplitPcaScale(WB)

    pre_WWB = Preprocess(WWB)
    X_train_WWB, X_test_WWB, y_train_WWB, y_test_WWB = pre_WWB.SplitPcaScale(WWB)


if __name__ == '__main__':
    startML()


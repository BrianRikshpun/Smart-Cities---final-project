import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.svm import SVR
#from tqdm.notebook import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import GridSearchCV
import math
from Preprocess import Preprocess


def startML():
    WB = pd.read_csv("WB.csv") #Water data
    WWB = pd.read_csv('WWB.csv') #Water and weather data

    pre_WB = Preprocess(WB)
    X_train_WB, X_test_WB, y_train_WB, y_test_WB = pre_WB.SplitPcaScale(WB)

    pre_WWB = Preprocess(WWB)
    X_train_WWB, X_test_WWB, y_train_WWB, y_test_WWB = pre_WWB.SplitPcaScale(WWB)

if __name__ == '__main__':
    startML()


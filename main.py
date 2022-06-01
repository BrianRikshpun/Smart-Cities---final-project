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
    WB = pd.read_csv("WB.csv")
    WWB = pd.read_csv('WWB.csv')

    pre = Preprocess(WB)
    X,x1 = pre.SplitPcaScale(WB)

    print(pd.DataFrame(X).info())
    print(pd.DataFrame(x1).info())

if __name__ == '__main__':
    startML()


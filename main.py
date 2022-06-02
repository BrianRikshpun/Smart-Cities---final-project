
import pandas as pd
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


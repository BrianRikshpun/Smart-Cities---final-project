import pandas as pd

def startML():
    WB = pd.read_csv("WB.csv")
    WWB = pd.read_csv('WWB.csv')
    WeatherAndWaterC = pd.read_csv('WeatherAndWaterDateC.csv')

    print(WB.head())
    print("------")

    print(WWB.head())
    print("------")

    print(WeatherAndWaterC.head())
    print("------")

if __name__ == '__main__':
    startML()


'''
This code contains all the preprocessing functions to load in and process data.

'''

#Import Libraries
import os # operating system module
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization and plotting
import seaborn as sns # statistical plotting
from datetime import datetime # Convert Datetime 
from sklearn.model_selection import TimeSeriesSplit, train_test_split # for linear model
from sklearn.preprocessing import StandardScaler #for scaling data



def get_data(f,st,sn):
    
    full_fp = (f + "\\" + st + "\\" + sn + ".us.txt")
    df = pd.read_csv(full_fp)
    
    return df

def del_OI(df_sorted):

    df_dropped = df_sorted.drop(labels=["OpenInt"], axis=1)
    
    return df_dropped

def std_vol(df_dropped):
    
    scaled_features = df_dropped.copy()
    col_names = ['Volume']
    features = scaled_features[col_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    scaled_features[col_names] = features
    
    
    df_scaled = scaled_features
    
    return df_scaled

def get_daily_deltas(df_scaled):

    df_scaled["High-Low"] = df_scaled["High"] - df_scaled["Low"]
    df_scaled["Close-Open"] = df_scaled["Close"] - df_scaled["Open"]

    return df_scaled

def nearendquarter(month,day):

    if month in [3, 6, 9, 12]:
        if month in [3, 12]:
            if day in [31, 30, 29, 28, 27, 26, 25]:
                return 1
            else:
                return 0
        else:
            if day in [30, 29, 28, 27, 26, 25, 24]:
                return 1
            else: 
                return 0
    else:
        return 0

def convertdatetime(s):
    
    date = pd.to_datetime(s)

    return date

def converttoday(dt):

    dt =  dt.weekday()

    return dt

def near_end_quart(df_added):

    splitted = df_added['Date'].str.split('-', expand=True)
 
    df_added['Day_date'] = splitted[2].astype('int')
    df_added['Month'] = splitted[1].astype('int')
    df_added['Year'] = splitted[0].astype('int')

    
    df_added['near_end_quarter'] = df_added.apply(lambda x: \
        nearendquarter(x['Month'], x['Day_date']), axis=1)
    
    
    #convert Date to datetime in order to day the day of the week
    df_added['Date'] = df_added['Date'].apply(convertdatetime)
    df_added['Day'] = df_added['Date'].apply(converttoday)

    #reindex by the date so it can be split into time series data later
    df_added.set_index('Date',inplace=True)
    df_added.sort_index(inplace=True)

    return df_added

def split_data(df_end_quart):

    X = df_end_quart.drop(labels=["Close"], axis=1)
    T = df_end_quart["Close"]

    tss = TimeSeriesSplit(n_splits = 2)

    for train_indx, test_indx in tss.split(X):
    
        X_train, X_test = X.iloc[train_indx, :], X.iloc[test_indx,:]
        T_train, T_test = T.iloc[train_indx], T.iloc[test_indx] 

    return X_train, X_test, T_train, T_test

def  market_prepro(f,st,sn,verbose=False):
    df = get_data(f,st,sn) #get the dataset imported 
    df_dropped = del_OI(df)
    df_scaled = std_vol(df_dropped)
    df_added = get_daily_deltas(df_scaled)
    df_end_quart = near_end_quart(df_added)
    X_train, X_test, T_train, T_test = split_data(df_end_quart)

    if verbose == True:
        print(df_end_quart.head())
        print(df_end_quart.info())
        print(df_end_quart.describe())
        print("The start date of the training data is ", (X_train[:1]))
        print("The last date of the training data is ", (X_train[-1:]))
        print("The start date of the training data is ", (X_test[:1]))
        print("The last date of the training data is ", (X_test[-1:]))

        sns.heatmap(df_end_quart.corr(),annot=True, square=True, cmap='terrain', linewidths=0.1)
        sns.pairplot(df_end_quart)
        plt.show()

    return X_train, X_test, T_train, T_test


def test():
    # st = "Stocks"
    st = "ETFs"

    #Input stock name
    sn = "aadr" 
    f = r'D:\Desktop\College Spring 2023\machineLearning\project\coding\data'
    X_train, X_test, T_train, T_test = market_prepro(f,st,sn,True)


    # print(X_train)

if __name__ == "__main__":
    test()

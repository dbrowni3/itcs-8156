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
from sklearn.preprocessing import MinMaxScaler #for scaling data

from torch.utils.data import TensorDataset, DataLoader
import torch




def get_data(f,st,sn):
    
    full_fp = (f + "\\" + st + "\\" + sn + ".us.txt")
    df = pd.read_csv(full_fp)
    
    return df

def del_OI(df_sorted):

    df_dropped = df_sorted.drop(labels=["OpenInt"], axis=1)
    
    return df_dropped

def std_vol(df_dropped, stdzr):
    
    scaled_features = df_dropped.copy()
    col_names = df_dropped.columns.values
    col_names = np.delete(col_names,np.argwhere(col_names=='Date'))

    # col_names = np.delete(col_names,np.argwhere(col_names=='Close'))
    # col_names = ['Volume']
    features = scaled_features[col_names]
    if (stdzr == 'standard'):
        scaler = StandardScaler().fit(features.values)
    elif (stdzr == 'minmax'):
        scaler = MinMaxScaler().fit(features.values)

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

def  market_prepro(f,st,sn, verbose=False, splitdata=True, stdzr='minmax'):
    df = get_data(f,st,sn) #get the dataset imported 
    df_dropped = del_OI(df)
    df_scaled = std_vol(df_dropped,stdzr)
    df_added = get_daily_deltas(df_scaled)
    df_end_quart = near_end_quart(df_added)


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

    if (splitdata == True):
        X_train, X_test, T_train, T_test = split_data(df_end_quart)
        return X_train, X_test, T_train, T_test
    else:
        X = df_end_quart.drop(labels=["Close"], axis=1)
        T = df_end_quart["Close"]
        return X, T


def structure_timeseries_features(df,offset_back, offset_for,exclude):
    '''
    This takes a dataframe and creates new columns that contain the data from
    previous days so that time series forecasting can occur.

    INPUTS:
        df - dataframe: input data

        offset_back - int: the number of days to go back. This creates this many
                        new columns

        offset_for - int: the number of days to go forwards. This creates this many
                        new columns

        exclude - list str: list of columns to exclude from the time series 
                            expansion

    OUTPUTS:
        df_out - dataframe: dataframe with new columns
    '''

    df_out = pd.DataFrame() 

    for cc in df.columns.values:
        if (cc not in exclude):
            for ii in range(offset_back):
                col_name = (cc+"_m"+str(ii+1))

                df_out[col_name] = df[cc].shift(ii+1)


    df_out = df_out.iloc[offset_back:-offset_for,:]

    return df_out

def structure_timeseries_targets(df,offset_back, offset_for):
    '''
    companion function to remove the first few days of targets to make sure
    the sizes match between features and targets. Also offset the data for
    multi day targets

    INPUTS:
        df - dataframe: input data targets.

        offset_back - int: the number of days to go back. This creates this many
                        new columns
                        
        offset_for - int: the number of days to go forwards. This creates this many
                        new columns

    OUTPUTS:
        df_out - dataframe: dataframe with rows removed

    '''
    # df_out = df.iloc[offset:]

    df_out = pd.DataFrame() 

    # for cc in df.columns.values:
    for ii in range(offset_for):
        col_name = (df.name+"_p"+str(ii))

        df_out[col_name] = df.shift(-(ii))


    df_out = df_out.iloc[offset_back:-offset_for,:]

    return df_out




def lstm_timeseries_feat_and_targ(df_feat, df_targ, offset_back, offset_for, exclude):
    '''
    This takes a dataframe and creates new columns that contain the data from
    previous days so that time series forecasting can occur.

    INPUTS:
        df - dataframe: input data

        offset_back - int: the number of days to go back. This include the current day
                            current day + (offset_back - 1 ) = offset_back

        offset_for - int: the number of days to go forwards. This creates this many
                        new columns

        exclude - list str: list of columns to exclude from the time series 
                            expansion

    OUTPUTS:
        dataloader - dataframe: dataframe with new columns
    '''
    if (exclude != None and isinstance(df_feat,pd.DataFrame)):
        df_feat = df_feat.drop(exclude, axis=1)



    #the offset number includes the current day so for an offset_back = 2
    #you only need the current day and one day back. So I am going to switch it to
    #offset_back = offset_back - 1

    #get the number of samples that will be created. trim off the ends that
    #can't be used
    num_samps = len(df_feat)-(offset_back-1)-offset_for
    indx_end = len(df_feat)
    # print('num samples ,' ,num_samps)
    # print('num samples raw ,' , len(df_feat))

    #get the number of features
    if isinstance(df_feat,pd.Series):
        num_feats = 1
    else:
        num_feats = len(df_feat.columns)

    #init the feature and target arrays
    features = np.zeros((num_samps, num_feats, offset_back))
    targets =  np.zeros((num_samps, offset_for+1))

    # print('features.shape ', features.shape)
    # print('targets.shape ', targets.shape)

    for ii, dd in enumerate(range(offset_back-1, indx_end-offset_for)):
  
        if isinstance(df_feat,pd.Series):
            feat_temp = df_feat.iloc[dd-offset_back+1:dd+1].to_numpy().T
        else:
            feat_temp = df_feat.iloc[dd-offset_back+1:dd+1,:].to_numpy().T

        if (offset_for == 0):
            targ_temp = df_targ.iloc[dd]
        else:
            targ_temp = df_targ.iloc[dd:dd+offset_for+1].to_numpy()

        features[ii, :, :] = feat_temp
        targets[ii, :] = targ_temp

    # print('dd = ',dd)
    # print('ii = ',ii)
            
    # transform numpy to torch
    features = torch.from_numpy(features)    
    targets = torch.from_numpy(targets) 
    # print('feat size ', features.shape)   
    # print('targ size ', targets.shape)   

    # add the data into a dataloader
    dataset = TensorDataset(features, targets)
    dataloader = DataLoader(dataset, num_workers=2)

    return dataloader, dataset


def test():
    # st = "Stocks"
    st = "ETFs"

    #Input stock name
    sn = "aadr" 
    f = r'D:\Desktop\College Spring 2023\machineLearning\project\coding\data'
    X_train, X_test, T_train, T_test = market_prepro(f,st,sn,False,splitdata=True)
    # X,T = market_prepro(f,st,sn,False,splitdata=False, stdzr='minmax')

    # dl, ds = lstm_timeseries_feat_and_targ(X, T, 3, 1,[ 'Year', 'Month' ,'Day_date', 'Day'])

    # dl_train, ds_train = lstm_timeseries_feat_and_targ(X_train, T_train, 6, 1, [ 'Year', 'Month' ,'Day_date', 'Day'])
    dl_test, ds_test = lstm_timeseries_feat_and_targ(X_test, T_test, 2, 0, [ 'Year', 'Month' ,'Day_date', 'Day'])

    # print(ds_test[-1])
    # print(ds_train[-6])




    
    # print(X_test.head(10))
    # print(T_test.head(10))
    # print(X_test.tail(10))
    # print(T_test.tail(10))
    # print(np.where(T == np.min(T)))

    # print(ds.shape)



if __name__ == "__main__":
    test()

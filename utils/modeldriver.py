import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_percentage_error,
                            )
from sklearn.linear_model import (Ridge,
                                  Lasso,
                                )
from preprocessing import (market_prepro,
                           structure_timeseries_features,
                           structure_timeseries_targets,
)



def modeldriver(mdl, n_days_back, n_days_for, f, st, sn):
    '''
    This function takes a model, imports the data, trains the model, and tests 
    the model

    INPUTS:
        mdl -  the model to test

        n_days_back - int: the number of days to go back to use as features

        n_days_for - int: the number of days to predict into the future

        f - str: filepath to the stock

        st - str: stock type

        sn - str: stock name
    OUTPUTS:
        NONE:
    '''

    exclude_col = ['Day_date','Month','Year']

    X_train, X_test, T_train, T_test = market_prepro(f,st,sn)

    X_train = structure_timeseries_features(X_train,
                                            n_days_back, 
                                            n_days_for, 
                                            exclude_col,
                                            )
    X_test = structure_timeseries_features(X_test,
                                           n_days_back, 
                                           n_days_for, 
                                           exclude_col,
                                           )
    T_train =  structure_timeseries_targets(T_train,
                                            n_days_back,
                                            n_days_for,
                                            )
    T_test =  structure_timeseries_targets(T_test,
                                            n_days_back,
                                            n_days_for,
                                            )
    
    data = {'X_train': X_train, 
            'X_test': X_test, 
            'T_train': T_train, 
            'T_test':T_test}
    

    mdl.fit(X_train,T_train)

    y_train = mdl.predict(X_train)
    y_test = mdl.predict(X_test)

    y_train = pd.DataFrame(y_train,
                           columns=T_train.columns.values,
                           index=T_train.index
                           )
    y_test = pd.DataFrame(y_test,
                          columns=T_test.columns.values,
                          index=T_test.index
                          )

    score_train = mdl.score(X_train,T_train)
    score_test = mdl.score(X_test,T_test)

    test_mape = mean_absolute_percentage_error(data['T_test'],y_test, multioutput='raw_values')

    print('Test MAPE(%) ', test_mape*100)

    return y_train, y_test, data



def test():

    # st = "Stocks"
    st = "ETFs"

    #Input stock name
    sn = "aadr" 
    f = r'D:\Desktop\College Spring 2023\machineLearning\project\coding\data'
    X_train, X_test, T_train, T_test = market_prepro(f,st,sn)

    print(X_train.info())


    n_days_back = 5
    n_days_for = 5

    print('Ridge')
    mdl_r = Ridge(alpha=1)

    y_train, y_test, data_out = modeldriver(mdl_r, n_days_back, n_days_for, f, st, sn)




if __name__ == "__main__":
    test()
    
from sklearn.model_selection import train_test_split


def rmse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error

    results = {}
    for column in y_true.columns:
        results[column] = mean_squared_error(y_true[column], 
                                             y_pred[column], 
                                             squared=False)
    
    return pd.Series(results)

def mape(y_true, y_pred):
    from sklearn.metrics import mean_absolute_percentage_error

    results = {}
    for column in y_true.columns:
        results[column] = mean_absolute_percentage_error(y_true[column], 
                                                         y_pred[column])
    
    return pd.Series(results)

def mase(y_true, y_pred, y_train):
    from metrics import mean_absolute_scaled_error

    return mean_absolute_scaled_error(y_true, y_pred, y_train)
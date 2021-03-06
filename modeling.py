import pickle
import matplotlib.pyplot as plt
from hts import HTSRegressor
from hts.hierarchy import HierarchyTree
from m5 import *
import time
import pickle
import time
import json
from sklearn.model_selection import train_test_split
from statistics import mean

import warnings
warnings.filterwarnings("ignore")

# Function to reduce memory to increase performance to decrease run time.
def downcasting(df, verbose=False):
    '''
    reduce memory usage by downcasting data types
    from https://www.kaggle.com/harupy/m5-baseline
    '''

    start_mem = df.memory_usage().sum() / 1024 ** 2
    int_columns = df.select_dtypes(include=["int"]).columns
    float_columns = df.select_dtypes(include=["float"]).columns

    for col in int_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def runModel(modelName, hierarchy, dataFrame, revisionMethod=None):

    clf = HTSRegressor(model=modelName, revision_method=revisionMethod, n_jobs=4)
    return clf.fit(dataFrame, hierarchy)


def prediction(model, steps=28):
    return model.predict(steps_ahead=steps)


def saveModel(path, model, method, ols, samples=None):

    dt = time.localtime()
    strinstante = "%02d-%02d-%04d-%02d-%02d-%02d" % (
    dt.tm_year, dt.tm_mon, dt.tm_mday, dt.tm_hour, dt.tm_min, dt.tm_sec)

    if (samples):
        filename = path + "model" + method + ols + str(samples) + strinstante + ".pkl"
    else:
        filename = path + "model" + method + ols + strinstante + ".pkl"

    with open(filename, 'wb') as file:
        pickle.dump(model, file, protocol=4)

        
def loadModel(path):
        
    with open(path, 'rb') as file:
        model = pickle.load(file)
        
    return model


def saveDataset(path, dataFrame, method, ols, samples=None):

    dt = time.localtime()
    strinstante = "%02d-%02d-%04d-%02d-%02d-%02d" % (
    dt.tm_year, dt.tm_mon, dt.tm_mday, dt.tm_hour, dt.tm_min, dt.tm_sec)

    if (samples):
        filename = path + "dataset" + method + ols + str(samples) + strinstante + ".pkl"
    else:
        filename = path + "dataset" + method + ols + strinstante + ".pkl"

    dataFrame.to_pickle(filename)

def plot(dictPlot, predictions, dataFrame):

    fig, axs = plt.subplots(len(dictPlot), figsize=(20, 30), sharex=True)
    # ax.grid(alpha=0.75)
    PLOT_FROM = 0
    dts = predictions[PLOT_FROM:].index

    for i, group in enumerate(dictPlot):
        axs[i].plot(dataFrame[PLOT_FROM:][group],
                lw=1.1,
                color='#2ecc71',
                alpha=0.8,
                label='Truth')
        axs[i].plot(predictions[PLOT_FROM:][group],
                lw=1.1,
                color='#e74c3c',
                alpha=0.8,
                label='Prediction')
        axs[i].grid(alpha=0.75)
        axs[i].legend()
        axs[i].set_title(group)


def createHierarchyTree(self, _last=True):

    ht_tree = {
            'name':  self.key,
        }
    child_count = len(self.children)

    for i, child in enumerate(self.children):
        _last = i == (child_count - 1)
        if i == 0:
            ht_tree['children'] = []

        ht_tree['children'].append(createHierarchyTree(child, _last=_last))
    return ht_tree


def erroBarGraf(dict_front_data):
    errorJson = {
            "legend":[
                      "RMSE",
                      "MAPE"
                   ],
             "name": [],
          
             "value":[
                      {
                         "RMSE":[],
                         "MAPE":[]
                      }
                   ]
        }
    for key in dict_front_data.keys():
        errorJson["name"].append(key)
        for k in dict_front_data[key].keys():
            errorJson["value"][0]['RMSE'].append(mean(list(dict_front_data[key][k]['RMSE'])))
            errorJson["value"][0]["MAPE"].append(mean(list(dict_front_data[key][k]['MAPE'])))
            
    
    return errorJson


def errorToJson(dict_data, revisionMethodName):
    errorRMSE = {"name": "RMSE",
                 "values":[]
                }
    
    errorMAPE = {"name": "MAPE",
                 "values":[]
                }
    errorRMSE["values"] = { key : list() for key in [revisionMethodName]} if isinstance(revisionMethodName, str) else { key : list() for key in revisionMethodName}
    errorMAPE["values"] = { key : list() for key in [revisionMethodName]} if isinstance(revisionMethodName, str) else { key : list() for key in revisionMethodName}
    
    for key in dict_data.keys():
        for k in dict_data[key].keys():
            errorRMSE["values"][k].append(mean(list(dict_data[key][k]['RMSE'])))
            errorMAPE["values"][k].append(mean(list(dict_data[key][k]['MAPE'])))
    
    return errorRMSE, errorMAPE


def jsonResult(self, modelsList, revisionMethodList, prediction, hist, error, _last=True):
    
        
    ht_tree = {
        'name':  self.key,
        'values': [],
    }

    ht_tree['values'] = ht_tree.fromkeys([modelsList, 'historical']) if isinstance(modelsList, str) else ht_tree.fromkeys(x) 
    ht_tree['values']['historical'] =  list(hist[self.key])
    if isinstance(modelsList, str):
        ht_tree['values'][modelsList] = ht_tree.fromkeys([revisionMethodList]) if isinstance(revisionMethodList, str) else ht_tree.fromkeys(revisionMethodList)
        for revisionKey in ht_tree['values'][modelsList].keys():
            ht_tree['values'][modelsList][revisionKey] = ht_tree.fromkeys(['prediction', 'error'])
            ht_tree['values'][modelsList][revisionKey]['prediction'] = list(prediction[modelsList][revisionKey][self.key])
            ht_tree['values'][modelsList][revisionKey]['error'] = ht_tree.fromkeys(error[modelsList][revisionKey].keys())
            for errorKey in ht_tree['values'][modelsList][revisionKey]['error'].keys():
                ht_tree['values'][modelsList][revisionKey]['error'][errorKey] = error[modelsList][revisionKey][errorKey][self.key]

    else:
        for modelKey in modelsList:
            ht_tree['values'][modelKey] = ht_tree.fromkeys([revisionMethodList]) if isinstance(revisionMethodList, str) else ht_tree.fromkeys(revisionMethodList)
            for revisionKey in ht_tree['values'][modelKey].keys():
                ht_tree['values'][modelKey][revisionKey] = ht_tree.fromkeys(['prediction', 'error'])
                ht_tree['values'][modelKey][revisionKey]['prediction'] = list(prediction[modelKey][revisionKey][self.key])
                ht_tree['values'][modelKey][revisionKey]['error'] = ht_tree.fromkeys(error[modelKey][revisionKey].keys())
                for errorKey in ht_tree['values'][modelKey][revisionKey]['error'].keys():
                    ht_tree['values'][modelKey][revisionKey]['error'][errorKey] = error[modelKey][revisionKey][errorKey][self.key]

    child_count = len(self.children)
    
    for i, child in enumerate(self.children):
        _last = i == (child_count - 1)
        if i == 0:
            ht_tree['nivel'] = []
            
        ht_tree['nivel'].append(jsonResult(child, modelsList, revisionMethodName, prediction, hist, error, _last=_last))
    return ht_tree


def jsonResultFront(self, modelsList, revisionMethodList, prediction, hist, _last=True):
    
        
    ht_tree = {
        'name':  self.key,
        'values': [],
    }
    ht_tree['values'].append(['historico', list(hist[self.key])])
    
    if isinstance(modelsList, str):
        for revisionKey in prediction[modelsList].keys():
            ht_tree['values'].append([modelsList + '-' + revisionKey,list(prediction[modelsList][revisionKey][self.key])]) 

    else:
        for modelKey in modelsList:
            for revisionKey in prediction[modelKey].keys():
                ht_tree['values'].append([modelKey + '-' + revisionKey,list(prediction[modelKey][revisionKey][self.key])])        
    
    child_count = len(self.children)
    
    for i, child in enumerate(self.children):
        _last = i == (child_count - 1)
        if i == 0:
            ht_tree['nivel'] = []
            
        ht_tree['nivel'].append(jsonResultFront(child, modelsList, revisionMethodName, prediction, hist, _last=_last))
    return ht_tree

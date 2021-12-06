import pickle
import matplotlib.pyplot as plt
from hts import HTSRegressor
from hts.hierarchy import HierarchyTree
from m5 import *
import time

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

    dt = time.localtime()s
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

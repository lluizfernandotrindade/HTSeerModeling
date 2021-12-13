from m5 import *
from modeling import *
from evaluationMetrics import *
import pickle

dataset_path = 'data/'
data_set = M5(dataset_path, samples=50)
dfSelected = downcasting(data_set.train_set)

with open('jsonFrontBack.json') as jsonfile:
    frontJson = json.load(jsonfile)
    
modelsListName = frontJson['modelo']
revisionMethodName = frontJson['tecnica']
previsionSteps = frontJson['previsao']
hierarchy = data_set.hierarchy
train, test = train_test_split(dfSelected, test_size=previsionSteps, shuffle=False)


errorListKeys = ['RMSE', 'MAPE']
dict_error = {}
dict_prediction = {}

dict_error =  dict_error.fromkeys([modelsListName]) if isinstance(modelsListName, str) else dict_error.fromkeys(modelsListName) 
dict_prediction =  dict_prediction.fromkeys([modelsListName]) if isinstance(modelsListName, str) else dict_prediction.fromkeys(modelsListName)

if isinstance(modelsListName, str):
    dict_error[modelsListName] =  dict_error.fromkeys([revisionMethodName]) if isinstance(revisionMethodName, str) else dict_error.fromkeys(revisionMethodName) 
    dict_prediction[modelsListName] =  dict_prediction.fromkeys([revisionMethodName]) if isinstance(revisionMethodName, str) else dict_prediction.fromkeys(revisionMethodName)
    for key in dict_prediction[modelsListName].keys():
        modelTrained = runModel(modelsListName, hierarchy, train, revisionMethod=key)
        preds = prediction(modelTrained, previsionSteps)
        dict_prediction[modelsListName][key] = preds[-previsionSteps:]
        y_train = train
        y_true = test
        y_pred = preds[-previsionSteps:]
        
        dict_error[modelsListName][key] = dict_error.fromkeys([errorListKeys]) if isinstance(errorListKeys, str) else dict_error.fromkeys(errorListKeys) 
        dict_error[modelsListName][key]['RMSE'] = rmse(y_true, y_pred)
        dict_error[modelsListName][key]['MAPE'] = mape(y_true, y_pred)
        
else: 
    for model in modelsListName:
        dict_error[model] =  dict_error.fromkeys([revisionMethodName]) if isinstance(revisionMethodName, str) else dict_error.fromkeys(revisionMethodName) 
        dict_prediction[model] =  dict_prediction.fromkeys([revisionMethodName]) if isinstance(revisionMethodName, str) else dict_prediction.fromkeys(revisionMethodName) 
        for key in dict_prediction[model].keys():
            modelTrained = runModel(model, hierarchy, train, revisionMethod=key)
            preds = prediction(modelTrained, previsionSteps)
            dict_prediction[model][key] = preds[-previsionSteps:]
            y_train = train
            y_true = test
            y_pred = preds[-previsionSteps:]

            dict_error[model][key] = dict_error.fromkeys([errorListKeys]) if isinstance(errorListKeys, str) else dict_error.fromkeys(errorListKeys) 
            dict_error[model][key]['RMSE'] = rmse(y_true, y_pred)
            dict_error[model][key]['MAPE'] = mape(y_true, y_pred)
            
with open('dictError.json','wb') as f:
    pickle.dump(dict_error, f, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('dictPrediction.json', 'wb') as f:
    pickle.dump(dict_prediction, f, protocol=pickle.HIGHEST_PROTOCOL) 
    
grafError = erroBarGraf(dict_error)

with open('grafBarError.json', 'w', encoding='utf-8') as f:
    json.dump(grafError, f, ensure_ascii=False, indent=4)
    
errorRMSE, errorMAPE = errorToJson(dict_error, revisionMethodName)

with open('errorRMSE.json', 'w', encoding='utf-8') as f:
    json.dump(errorRMSE, f, ensure_ascii=False, indent=4)
    
with open('errorMAPE.json', 'w', encoding='utf-8') as f:
    json.dump(errorMAPE, f, ensure_ascii=False, indent=4)

finalJson = {
     "recorte": list( y_true.index.strftime("%Y-%m-%d")),
     "nivel": [jsonResultFront(ht, modelsListName, revisionMethodName, dict_prediction, y_true)]
 }

with open('result.json', 'w', encoding='utf-8') as f:
    json.dump(finalJson, f, ensure_ascii=False, indent=4)
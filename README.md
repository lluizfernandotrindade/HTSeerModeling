# Getting Started

This project consists of the modeling for the App HTSeer.


## Dataset M5 Forecasting

How much camping gear will one store sell each month in a year? To the uninitiated, calculating sales at this level may seem as difficult as predicting the weather. Both types of forecasting rely on science and historical data. While a wrong weather forecast may result in you carrying around an umbrella on a sunny day, inaccurate business forecasts could result in actual or opportunity losses. In this competition, in addition to traditional forecasting methods you’re also challenged to use machine learning to improve forecast accuracy.

The Makridakis Open Forecasting Center (MOFC) at the University of Nicosia conducts cutting-edge forecasting research and provides business forecast training. It helps companies achieve accurate predictions, estimate the levels of uncertainty, avoiding costly mistakes, and apply best forecasting practices. The MOFC is well known for its Makridakis Competitions, the first of which ran in the 1980s.

In this competition, the fifth iteration, you will use hierarchical sales data from Walmart, the world’s largest company by revenue, to forecast daily sales for the next 28 days. The data, covers stores in three US States (California, Texas, and Wisconsin) and includes item level, department, product categories, and store details. In addition, it has explanatory variables such as price, promotions, day of the week, and special events. Together, this robust dataset can be used to improve forecasting accuracy.

If successful, your work will continue to advance the theory and practice of forecasting. The methods used can be applied in various business areas, such as setting up appropriate inventory or service levels. Through its business support and training, the MOFC will help distribute the tools and knowledge so others can achieve more accurate and better calibrated forecasts, reduce waste and be able to appreciate uncertainty and its risk implications.

Dataset is Available in: [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy).

## Available Scripts

In the project directory, you can run:

### `m5.py`

Script responsible for loading data from M5, as well as creating the hierarchy that was used by the modeling algorithms in this project.The created class has the properties:

- self.path = Server dataset path
- self.calendar = Calendar with holidays
- self.train_set = Data Frame selected for training
- self.sampleSize = Number of samples in the series

### `modeling.py`

Main run file with functions to manipulate and run the models. The main functions can be describe below:

```python
  def downcasting(df, verbose=False)
```
(Optional) Preprocess to reduce data size mem for best performance.

```python
  def createHierarchyTree(self, _last=True)
```
Function to create the hierarchy structure for front rendering.

```python
  def runModel(modelName, hierarchy, dataFrame, revisionMethod=None):
```
Main function to run the possible models provide by the lib [Scikit-hts](https://scikit-hts.readthedocs.io/en/latest/readme.html).

```python
  def prediction(model, steps=28):
```
Function to make the prediction after the model is training.

```python
  def saveModel(path, model, method, ols, samples=None):
```
Function to save the model after the training is finished. The properties required are:

  - path = server path where the model will be save.
  - model = the model object provide by the Scikit-hts lib.
  - method = the string method name.
  - ols = reconciliation string.
  - samples = number of samples used.

```python
  def loadModel(path):
```
Function to restore the model.

```python
  def saveDataset(path, dataFrame, method, ols, samples=None):
```
Function to save the dataset. The properties required are:

- path = server path where the model will be save.
- dataFrame = the data frame used to train/validate the model.
- method = the string method name.
- ols = reconciliation string.
- samples = number of samples used.

```python
  def plot(dictPlot, predictions, dataFrame):
```
Function to plot the result. The properties required are:

- dictPlot = the data frame header.
- predictions = the data frame predicted by the model.
- dataFrame = the data frame used as ground Through.

## Modeling Strategy

The model Strategy can be found with more detail in this link [scikit-hts examples](https://github.com/carlomazzaferro/scikit-hts-examples/blob/master/notebooks/M5.ipynb).

### HTSeer Modeling

Showcase for more info how the algorithms works can be found in the jupyter notebook HTSeer Modeling.ipynb.

## Reference

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `Future Works`

For more sophisticated modeling, there is also this reference that can be used as a basis for comparison and study for future work. [M5-methods](https://github.com/Mcompetitions/M5-methods)

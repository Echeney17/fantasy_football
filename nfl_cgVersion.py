#Camerons update
#4/19/2021
#Changed accuracy to use preds as second parameter

pip install optuna

import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor, plot_tree, plot_importance
import sklearn
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz
import numpy as np
import optuna
import functools
import math
from google.colab import files
import io

data_to_load = files.upload()

nfl = pd.read_csv('nfl_contest.csv')
nfl.head()

X = nfl.iloc[:, 1:13]
Y = nfl.iloc[:, 13]

Y.head()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.30,random_state=123)

def objective(trial):
    train_x, valid_x, train_y, valid_y = train_test_split(X, Y, test_size=0.3)
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    bst = xgb.train(param, dtrain)
    preds = bst.predict(dvalid)
    pred_labels = np.rint(preds)
    accuracy = sklearn.metrics.mean_squared_error(valid_y,preds)
    return accuracy

study = optuna.create_study()
study.optimize(objective, n_trials=100)

study.best_params

def getBestValue(int):
  X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=(int/100),random_state=123)
  study = optuna.create_study(direction='minimize')
  study.optimize(objective, n_trials=10)
  return study.best_value

results=[]
input=[]
for number in range(1,100):
  results.append(getBestValue(number))
  input.append(number)

plt.plot(input,results)
plt.title('Fantasy Football Predictions')
plt.xlabel('Percent of Data Trained')
plt.ylabel('Error Rate Percentage')
plt.show()

alg = xgb.XGBRegressor(colsample_bytree=.7,learning_rate=.07,max_depth=2,min_child_weight=15,n_estimators=72,subsample=.9)
algs = alg.fit(X_train,Y_train)
y_pred = algs.predict(X_test)
print("RMSE: " + str(math.sqrt(mean_squared_error(Y_test,y_pred))))
print("R2: " + str(r2_score(Y_test,y_pred)))

plot_importance(algs)

plot_tree(algs)

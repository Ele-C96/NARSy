from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    median_absolute_error
)
import numpy as np


def classification_metrics(y_true, y_pred, y_prob):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    if y_true.ndim > 1:
        y_true = y_true.argmax(axis=1)
    if y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_prob = y_prob.reshape(-1)

    m = {}

    m["accuracy"] = accuracy_score(y_true, y_pred)
    m["precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    m["recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    m["f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    try:
        if y_prob.ndim == 1:
            m["auc"] = roc_auc_score(y_true, y_prob)
        elif y_prob.shape[1] == 2:
            m["auc"] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            m["auc"] = roc_auc_score(y_true, y_prob, multi_class="ovr")
    except:
        m["auc"] = None

    return m


def regression_metrics(y_true, y_pred):

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    m = {}

    m["mae"] = mean_absolute_error(y_true, y_pred)
    m["mse"] = mean_squared_error(y_true, y_pred)
    m["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
    m["r2"] = r2_score(y_true, y_pred)
    m["median_ae"] = median_absolute_error(y_true, y_pred)

   
    if np.all(y_true != 0):
        m["mape"] = np.mean(np.abs((y_true - y_pred)/y_true))

    return m

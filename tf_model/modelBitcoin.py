import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential 
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from collections import deque
import math 
import os
import time
import numpy as np
import pandas as pd
import random
import sys
sys.path.append('../')
from tf_model.tcn import TCN, tcn_full_summary
from tf_model.params import *


def getScaler(df, dfcolumns):
    column_scaler = {}
    for column in dfcolumns:
        scaler = preprocessing.MinMaxScaler()
        df[column] = scaler.fit_transform(
            np.expand_dims(df[column].values, axis=1))
        column_scaler[column] = scaler
    return column_scaler

def readData(wsize=1, lookup=1, tsize=1, dfcolumns=[], shuffle=True):

    df = pd.read_csv("../data/BTC-USD.csv")
    df.dropna(inplace=True)

    result = {}
    result['df'] = df.copy()

    result["minmaxScaler"] = getScaler(df, dfcolumns)
    df['future'] = df['Adj Close'].shift(-lookup)
    forPrediction = np.array(df[dfcolumns].tail(lookup))
    
    df.dropna(inplace=True)

    allData = []
    windowData = deque(maxlen=wsize)

    for val, target in zip(df[dfcolumns].values, df['future'].values):
        windowData.append(val)
        if len(windowData) == wsize:
            allData.append([np.array(windowData), target])

    forPrediction = list(windowData) + list(forPrediction)
    forPrediction = list(forPrediction)
    forPrediction = np.array(forPrediction)
    result['forPrediction'] = forPrediction

    X, y = [], []
    for seq, target in allData:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape((X.shape[0], X.shape[2], X.shape[1]))
    result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=tsize, 
                                                                                shuffle=shuffle)

    return result

def createModel(dfcolumns, wsize):  
    # 3D tensor with shape (batch_size, timesteps, input_dim)
    i = Input(batch_shape=(None, len(dfcolumns), wsize))
    o = TCN(padding='same', return_sequences=True)(i)
    o = TCN(padding='same', return_sequences=False)(o)
    o = Dense(1, activation="linear")(o)
    m = Model(inputs=[i], outputs=[o])

    #m.summary()
    return m

def buildModel(data, m, batchSize, ep, opt, ls, debug=False):
    m.compile(optimizer=opt,
              metrics=["mean_absolute_error"],
              loss=ls)

    checkpointer = ModelCheckpoint("results", save_weights_only=True, save_best_only=True, verbose=0)
    tensorboard = TensorBoard(log_dir="logs")

    history = m.fit(data["X_train"], data["y_train"],
                    batch_size=batchSize,
                    epochs=ep,
                    validation_data=(data["X_test"], data["y_test"]),
                    callbacks=[checkpointer, tensorboard],
                    verbose=1)

    if debug:
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

                    

def plotFullPrediction(model, data):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["minmaxScaler"]["Adj Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["minmaxScaler"]["Adj Close"].inverse_transform(y_pred))
    plt.plot(y_test[-14:], c='b')
    plt.plot(y_pred[-14:], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Price", "Predicted"])
    plt.show()


def processEWMA(data, window):
    alpha = 2 /(window + 1.0)
    alpha_rev = 1-alpha

    scale = 1/alpha_rev
    n = data.shape[0]

    r = np.arange(n)
    scale_arr = scale**r
    offset = data[0]*alpha_rev**(r+1)
    pw0 = alpha*alpha_rev**(n-1)

    mult = data*pw0*scale_arr
    cumsums = mult.cumsum()
    result = offset + cumsums*scale_arr[::-1]
    return result


def getSignals(shortEMA, middleEMA, longEMA, allData, frame):
    buy = []
    sell = []
    buyFlag = 0
    sellFlag = 0

    for i in range(0, frame):
        if sellFlag==0 and shortEMA[i] < middleEMA[i] < longEMA[i]:
            sell.append([i,allData[i]])
            sellFlag = 1
            buyFlag = 0
        
        if buyFlag==0 and shortEMA[i] > middleEMA[i] > longEMA[i]:
            buy.append([i,allData[i]])
            buyFlag = 1
            sellFlag = 0
    
    return (buy, sell)


def plotPrediction(model, data, value):
    frame = 50
    y_test = data["y_test"]
    y_test = np.squeeze(data["minmaxScaler"]["Adj Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    
    partialTest = y_test[-frame:]
    partialTest = np.append(partialTest, value)
    buy = []
    sell = []

    shortEMA  = processEWMA(partialTest, 5)
    middleEMA = processEWMA(partialTest, 12)
    longEMA   = processEWMA(partialTest, 25)

    buy , sell = getSignals(shortEMA, middleEMA, longEMA, partialTest, frame)
    
    plt.plot(partialTest, c='b', marker=".", markersize=6)
    plt.plot(len(partialTest)-1, value, c='r', marker=".", markersize=20)
    plt.plot(shortEMA,  c='y', markersize=5)
    plt.plot(middleEMA, c='b', markersize=5)
    plt.plot(longEMA,   c='g', markersize=5)
    for i in buy:
        plt.plot(i[0],i[1], c='g', marker="^", markersize=10)
    for i in sell:
        plt.plot(i[0],i[1], c='r', marker="v", markersize=10)    
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Price", "Predicted"])
    plt.title('- 50 days timeframe -')
    plt.show()


def getAccuracy(model, data, lookup):
    y_test = data["y_test"]
    X_test = data["X_test"]
    y_pred = model.predict(X_test)
    y_test = np.squeeze(data["minmaxScaler"]["Adj Close"].inverse_transform(np.expand_dims(y_test, axis=0)))
    y_pred = np.squeeze(data["minmaxScaler"]["Adj Close"].inverse_transform(y_pred))
    y_pred = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-lookup], y_pred[lookup:]))
    y_test = list(map(lambda current, future: int(float(future) > float(current)), y_test[:-lookup], y_test[lookup:]))
    return accuracy_score(y_test, y_pred)


def predictPrice(model, data):
    forPrediction = data["forPrediction"][-WSIZE:]
    scalerCol = data["minmaxScaler"]
    forPrediction = forPrediction.reshape((forPrediction.shape[1], forPrediction.shape[0]))
    forPrediction = np.expand_dims(forPrediction, axis=0)
    prediction = model.predict(forPrediction)
    predictedPrice = scalerCol["Adj Close"].inverse_transform(prediction)[0][0]
    return predictedPrice

import pandas
import csv
from collections import defaultdict
from collections import namedtuple
import itertools
import math
import numpy as np
import pandas
import random
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def f(userid, sessions, train, Example):
    if train:
        object_train = Example(userID = userid, history = sessions[:-2], inputs = sessions[-2], target = sessions[-2][1:])
        return object_train
    else:
        return Example(userID = userid, history = sessions[:-1], inputs = sessions[-1], target = sessions[-1][1:])

def createExamples(userBase):
    ''' Create training and testing set '''

    # A simple way to define a class is using namedtuple.
    Example = namedtuple("Example", ["userID", "history", "inputs", "target"])

    userBase = pandas.DataFrame(userBase)

    userBase.reset_index(level = 0, inplace = True)

    trainData = userBase.apply(lambda x: f(x['USERID'], x['PRODUCTID'], True, Example), axis = 1).tolist()

    testData = userBase.apply(lambda x: f(x['USERID'], x['PRODUCTID'], False, Example), axis = 1).tolist()

    return trainData, testData

def getTrainTest(data):
    '''Divide data into train and test set'''

    data['SESSION'] = pandas.to_datetime(data['TIMESTAMP'],unit='s').dt.date

    # Print some statistics
    print("Average number of sessions per user")
    print(data.groupby('USERID')['SESSION'].nunique().mean())

    print("Average number of clicks per session")
    print(data.groupby(['USERID', 'SESSION'])['ACTION'].count().mean())

    # Create nested list of sessions and items per user
    userBase = data.groupby(['USERID', 'SESSION'])['PRODUCTID'].apply(list).groupby('USERID').apply(list)

    trainData, testData = createExamples(userBase)

    return trainData, testData
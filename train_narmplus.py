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
import helpers as h

from NarmPlus import NarmPlus

def get_examples(data, shuffle=True, **kwargs):
    """Shuffle data set and return 1 example at a time (until nothing left)"""
    if shuffle:
        random.shuffle(data)  # shuffle training data each epoch
    for example in data:
        yield example

def get_minibatch(data, batch_size=25, shuffle=True):
    """Return minibatches, optional shuffling""" 
    if shuffle:
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []    
        # in case there is something left
        if len(batch) > 0:
            yield batch

def prepare_example(example):
    """
    Turn an example into tensors of inputs and target.
    """
    v = torch.FloatTensor([example.userID])
    v = v.to(device)
    
    w = torch.FloatTensor([example.history])
    w = w.to(device)
    
    x = torch.FloatTensor([example.inputs])
    x = x.to(device)

    y = torch.FloatTensor([example.target])
    y = y.to(device)

    return v, w, x, y


def main(dataset):

    sample = pandas.read_pickle(dataset)

    trainData, testData = h.getTrainTest(sample)


if __name__ == '__main__':

    dataset = "./data/smallTaoBao.pkl"

    main(dataset)






    



# HELPER FUNCTIONS


# function to yield a (mini-)batch
def get_minibatch(data, batch_size=25, shuffle=True):
    """Return minibatches, optional shuffling""" 
    if shuffle:
#         print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []

    # yield minibatches
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []    
        # in case there is something left
        if len(batch) > 0:
            yield batch

# function to make the (mini-)batch ready for usage by the model
def prepare_minibatch(mb):
    """
    Minibatch is a list of examples.
    This function converts returns
    torch tensors to be used as input/targets.
    """
    batch_size = len(mb)
    x = [ex.inputs for ex in mb]
    x = torch.FloatTensor(x)
    x = x.to(device)

    y = [ex.target for ex in mb]
    y = torch.FloatTensor(y)
    y = y.to(device)

    return x, y

# simple evaluation function
def simple_evaluate(model, data, prep_fn=prepare_example, **kwargs):
    """Explained Variance Score of a model on given data set."""
    model.eval()  # disable dropout (explained later)
    targets = []
    predictions = []

    for example in data:

        # convert the example input and label to PyTorch tensors
        targets.append(example.target)
        x, target = prepare_example(example)

        # forward pass
        # get the output from the neural network for input x
        with torch.no_grad():
            output = model(x)
        # output shape: (batch, output_size)
        prediction = output[0].tolist()
        predictions.append(prediction)
            
    score = explained_variance_score(targets, predictions, multioutput='variance_weighted')

    return score, None


In[124]:



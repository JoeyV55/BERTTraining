import pandas as pd
import numpy as np
import csv 
import string
import nltk 
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

import time
import copy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
from os import path
from sklearn.metrics import hamming_loss, accuracy_score, roc_curve, auc, roc_auc_score, f1_score, multilabel_confusion_matrix, precision_recall_fscore_support

from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
from transformers import BertTokenizer
from pathlib import Path
import torch
from torch import Tensor
from box import Box
import pandas as pd
import collections
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
import apex
from sklearn.model_selection import train_test_split

import datetime

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc, F1, Hamming_loss

threshold = 0.5

def choose_labels(proj_name):
    labels = []
    if proj_name == "jabRef_2":
        
        #someclean_SMOTE, unclean_SMOTE
        labels = ["Util","NLP","APM","Network","DB","Interpreter","Logging","Data.Structure","i18n","DevOps",
        "Logic","Microservices","Test","Search","IO","UI","Parser","Security","App"]
    elif proj_name == "guava_2":
        #Nofilter: someClean_NOSMOTE, UNCLEAN
        #Filter: 
        """ labels = ["Util",
        "NLP",
        "APM",
        "Network",
        "DB",
        "Interpreter",
        "Logging",
        "DataStructure",
        "i18n",
        "DevOps",
        "Logic",
        "Microservices",
        "ML",
        "Test",
        "Search",
        "IO",
        "UI",
        "Parser",
        "Security",
        "Cloud",
        "BigData",
        "App",
        "GIS"]"""
        #someCleanSmote, uncleanSMOTE
        labels = ["Util","Network","DB","Interpreter","i18n","Logic","Test","IO","UI","Security","App"]

    elif proj_name == "mockito_2":
        #someClean_NOSMOTE? Everything else besides SMOTE?
        """ labels = ["Util", "NLP", "APM", "Network", "DB","Interpreter", "Logging", "DataStructure", 
        "i18n","DevOps", "Logic", "Microservices", "ML", "Test", "Search", "IO", "UI", 
        "Parser", "Security", "Cloud", "BigData", "App", "GIS"]"""
        
        #someClean_SMOTE
        labels = ["Util","Network","Interpreter","i18n","Logic","Test","IO","Security"]
    elif proj_name == "rxjava_2":
        #someClean_NOSMOTE?
        #labels = ["Util", "NLP", "APM", "Network", "DB", "Interpreter", "Logging", "DataStructure", "i18n", "DevOps", "Logic", "Microservices", "ML",
        # "Test", "Search", "IO", "UI", "Parser", "Security", "Cloud", "BigData", "App", "GIS"]
        
        #SomeCLean_SMOTE, UNCLEAN_SMOTE
        #Normal labels
        labels = ["Util","Network","Interpreter","i18n","Logic","Test","IO","App"]
        #Letter Labels
        #labels = ["A", "B", "C", "D", "E", "F","G","H"]
        #Full word labels
        #labels = ["Utility", "Network", "Interpreter","Internationalization","Logic","Test","Input and Output","Application"]
    elif proj_name == "audacity":
        labels = ["Util","APM","Network","DB","Error.Handling","Logging",
        "Lang","Data.Structure","i18n","Setup","Logic","IO","UI","Parser",
        "Event.Handling","App","GIS","Multimedia","CG"]
    elif proj_name == "powertoys":
        labels = ["APM","Interpreter","Logging","Data.Structure","i18n","Setup","Logic","Microservices",
        "Test","Search","UI","Parser","App"]
    elif proj_name == "powertoys_rmca":
        labels = ["APM","Interpreter","Logging","Data.Structure","i18n","Setup","Logic","Microservices",
        "Test","Search","UI","Parser","App"]
    elif proj_name == "rmca":
        labels = ["APM","Network","DB","Error.Handling","Logging","Lang","Data.Structure","DevOps","i18n","Setup","Microservices","UI","Parser","Security","Big.Data","Event.Handling"]
    elif proj_name == "all_2":
        labels = ["Util","NLP","APM","Network","DB","Interpreter","Error.Handling","Logging","Lang","Data.Structure","DevOps","i18n",
        "Setup","Logic","Microservices","Test","Search","IO","UI","Parser","Security","Big.Data","Event.Handling","App","Multimedia"]
    print("Labels selected: ", labels)
    return labels

def prec_rec_fscore_support_wrapper(y_pred:Tensor, y_true:Tensor, sigmoid:bool = True, thresh:float = threshold, average = "micro", sample_weight = None, **kwargs):
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return precision_recall_fscore_support(y_true, y_pred, average=average)

def F1_micro(y_pred:Tensor, y_true:Tensor, sigmoid:bool = True, thresh:float = threshold, average = "micro", sample_weight = None, **kwargs):
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average = average, sample_weight = sample_weight)



def Hamming_loss_mod(
    y_pred: Tensor,
    y_true: Tensor,
    sigmoid: bool = True,
    thresh: float = 0.5,
    sample_weight=None,
    **kwargs):
    if sigmoid:
        y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()
    return hamming_loss(y_true, y_pred, sample_weight=sample_weight)





"""
Used to find the total True Positives, False Positives, and False Negatives for use in computing the micro fmeasure.

    "micro":
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
"""
def findTPFNFP(confMatrixList):
    totalTP = 0
    totalFP = 0
    totalFN = 0
    for matrix in confMatrixList:
        print("First matrix: ", matrix)
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
from fast_bert.prediction import BertClassificationPredictor

MODEL_PATH = 'output/model_out'

def predictJabRefLabels():
    print("Running JabRef prediction")

def predictCombinedProjLabels():
    print("Running Bert with all model.")
    LABEL_PATH = './UNFILTERED/UNCLEAN_SMOTE/allBert/data/'
    
    predictor = BertClassificationPredictor(
                model_path=MODEL_PATH,
                label_path=LABEL_PATH, # location for labels.csv file
                multi_label=False,
                model_type='bert',
                do_lower_case=False,
                device=None) # set custom torch.device, defaults to cuda if available

    # Single prediction
    single_prediction = predictor.predict("just get me result for this text")
    print(single_prediction)

    # Batch predictions
    texts = [
        "this is the first text",
        "this is the second text"
        ]

    multiple_predictions = predictor.predict_batch(texts)
    print(multiple_predictions)
        

def runBertPredictions(proj_name):
    if(proj_name == "jabRef"):
        predictJabRefLabels()
    elif(proj_name == "all"):
        predictCombinedProjLabels()

runBertPredictions("all")




#!/usr/bin/env python
# coding: utf-8
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
import os
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
from os import path
from sklearn.metrics import hamming_loss, accuracy_score, roc_curve, auc, roc_auc_score, f1_score, multilabel_confusion_matrix

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

#threshold:float = 0.5 # Best keep it in [0.0, 1.0] range
#labels = ["Util", "NLP", "APM", "Network", "DB", "Interpreter", "Error Handling", "Logging", "Lang", 
#    "Data Structure", "DevOps", "i18n", "Setup", "Logic", "Microservices", "ML", "Test", "Search",
#    "IO", "UI", "Parser", "Security", "Cloud", "Big Data", "Event Handling", "App", "GIS"]

#Labels for rxJava
hypothesis = "Baseline"
proj_name = "rxjava_2"


if hypothesis == "Baseline":
    config = "issueTitle"
elif hypothesis == "H1":
    config = "issueBody"
elif hypothesis == "H2":
    config = "issueTitle + issueBody"
elif hypothesis == "H3":
    config = "issueTitle + issueBody + prComments"

print("Running project: ", proj_name, " with Hypothesis: ", hypothesis, "( ", config, " )")
labels = ["Util", "Network", "Interpreter", "i18n", "Logic", "Test", "IO", "App"]

threshold = 0.5

def F1_macro(y_pred:Tensor, y_true:Tensor, sigmoid:bool = True, thresh:float = threshold, average = 'macro', sample_weight = None, **kwargs):
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average = average, sample_weight = sample_weight)

def F1_by_label(y_pred:Tensor, y_true:Tensor, sigmoid:bool = True, thresh:float = threshold, sample_weight = None, labels:list = labels, **kwargs):
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return f1_score(y_true, y_pred, average = None)


def confusion_matrix_by_label(y_pred:Tensor, y_true:Tensor, sigmoid:bool = True, thresh:float = 0.5, sample_weight = None, samplewise = False, labels:list = labels, **kwargs):
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return multilabel_confusion_matrix(y_true, y_pred, labels = [i for i in range(len(labels))], sample_weight = sample_weight, samplewise = samplewise)

def accuracy_by_label(y_pred: Tensor, y_true: Tensor, sigmoid:bool = True, thresh:float = threshold, normalize:bool = True, sample_weight = None, labels:list = labels, **kwargs):
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    accuracies = {}
    for i in range(len(labels)):
        accuracies[labels[i]] = accuracy_score(y_true[:, i], y_pred[:, i], normalize = normalize, sample_weight = sample_weight)
    return accuracies

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

def findHammingLossByLabel(accuracyDict):
    hammingLossDict = {}
    for label in labels:
        hammingLossDict[label] = 1-accuracyDict[label]
    return hammingLossDict

def findRecallByLabel(confMatrixList):
    recallDict = {}
    index = 0
    for label in labels:
        currMatrix = confMatrixList[index]
        TP = currMatrix[1][1]
        FN = currMatrix[1][0]
        recallDict[label] = findRecall(TP, FN)
        index += 1
    return recallDict

def findPrecisionByLabel(confMatrixList):
    precisionDict = {}
    index = 0
    for label in labels:
        currMatrix = confMatrixList[index]
        TP = currMatrix[1][1]
        FP = currMatrix[0][1]
        precisionDict[label] = findPrecision(TP, FP)
        index += 1
    return precisionDict


def findFmeasure(TP, FP, FN):
    return TP/(TP + .5*(FP+FN))

def findPrecision(TP, FP):
    if(TP + FP) == 0:
        return 0.0
    return TP/(TP+FP)

def findRecall(TP, FN):
    if(TP + FN) == 0:
        return 0.0
    return TP/(TP+FN)

#Computes the overall metric score for validation of by label values
def sanityCheck(metricDict):
    total = 0
    for label in labels:
        total += metricDict[label]
    return total / len(labels)

def averageNonZeros(metricDict):
    total = 0
    count = 0
    for label in labels:
        if(metricDict[label] == 0.0):
            continue
        count+=1
        total += metricDict[label]
    return total / count




print("Started fast-bert, emptying cache")
torch.cuda.empty_cache()


DATA_PATH = './data/'
LABEL_PATH = './data/'
OUTPUT_PATH = './output/'



print("Building Box")
testSize = 0.8
#Build train and test csvs.
df = pd.read_csv('./data/binaryNew.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= testSize

train = df[msk]
test = df[~msk]
del train["split"]
del test["split"]
#del train["issueNumber"]
#del test["issueNumber"]
del train["prNumber"]
del test["prNumber"]

#train["corpus"] = train["corpus"].str.replace(r' +', ' ')
#test["corpus"] = test["corpus"].str.replace(r' +', ' ')
print(train.head())
print(test.head())
#print(type(train["issueNumber"][1]))
train.to_csv('./data/binaryTrain.csv', index=False)
test.to_csv('./data/binaryTest.csv', index=False)

args = Box({
    "run_text": "multilabel toxic comments with freezable layers",
    "train_size": -1,
    "val_size": -1,
    "log_path": OUTPUT_PATH,
    "full_data_dir": DATA_PATH,
    "data_dir": DATA_PATH,
    "task_name": "toxic_classification_lib",
    "no_cuda": False,
    "output_dir": OUTPUT_PATH,
    "max_seq_length": 512,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "learning_rate": 5e-5,
    "num_train_epochs": 6,
    "warmup_proportion": 0.0,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": True,
    "fp16_opt_level": "O1",
    "weight_decay": 0.0,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "max_steps": -1,
    "warmup_steps": 500,
    "logging_steps": 50,
    "eval_all_checkpoints": True,
    "overwrite_output_dir": True,
    "overwrite_cache": False,
    "seed": 42,
    "loss_scale": 128,
    "task_name": 'intent',
    "model_name": 'xlnet-base-cased',
    "model_type": 'xlnet'
})

databunch = BertDataBunch(DATA_PATH, LABEL_PATH,
                          tokenizer='bert-base-uncased',
                          train_file='binaryTrain.csv',
                          val_file='binaryTest.csv',
                          label_file='labelsNew.csv',
                          text_col='corpus',
                          label_col=labels,
                          batch_size_per_gpu=16,
                          max_seq_length=512,
                          multi_gpu=False,
                          multi_label=True,
                          model_type='bert')
print("Built databunch")

logger = logging.getLogger()
device_cuda = torch.device("cuda")
#,{'name': 'Hamming_loss', 'function': Hamming_loss_mod}
metrics = [{'name': 'accuracy', 'function': accuracy}, {'name': 'f1', 'function': F1}, {'name': 'Hammingloss', 'function': Hamming_loss_mod}, {'name': 'f1macro', 'function': F1_macro},
{'name': 'confmatrix', 'function': confusion_matrix_by_label},{'name': 'f1bylabel', 'function': F1_by_label},{'name': 'accuracybylabel', 'function': accuracy_by_label}]

OUTPUT_DIR = './output/'

print("Got cuda device")

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_PATH,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=False,
						is_fp16=True,
						multi_label=True,
						logging_steps=50)


print("Built learner")

learner.lr_find(start_lr=1e-5,optimizer_type='lamb')

print("Lr find")

#TODO Change back to 11 epochs or mess with it. 
learner.fit(epochs=21,
			lr=6e-5,
			validate=False, 	# Evaluate the model after each epoch
			return_results=True,
            schedule_type="warmup_cosine",
			optimizer_type="lamb")

print("Lr fit finished")

#Get the return with the results of this prediction
resultsVals = learner.validate()

print(resultsVals)


#Get array for confusion matrix
confMatrixList = resultsVals["confmatrix"]

recallDict = findRecallByLabel(confMatrixList)
precisionDict = findPrecisionByLabel(confMatrixList)

accuracyDict = resultsVals["accuracybylabel"]
hamLossDict = findHammingLossByLabel(accuracyDict)

recallAvg = sanityCheck(recallDict)
precisionAvg = sanityCheck(precisionDict)

accuracyScore = 1-resultsVals["Hammingloss"]


print("\n==================\n\nMetrics by label\n============================")
print("\nRecall by label: ", recallDict)
print("\nPrecision by label: ", precisionDict)
print("\HamLoss by label: ", hamLossDict)

print("\n==================\n\nSanity Check Averages (0s included)\n============================")
print("Hamming loss sanity check average: ", sanityCheck(hamLossDict))
print("Recall sanity check average: ", sanityCheck(recallDict))
print("Precision loss sanity check average: ", sanityCheck(precisionDict))


print("\n==================\n\nNonZero averages\n============================")
print("Recall skipping 0s", averageNonZeros(recallDict))
print("Precision skipping 0s", averageNonZeros(precisionDict))

learner.save_model()

dataDict = {"Hypothesis" : [hypothesis], "Precision" : [precisionAvg], "Recall": [recallAvg], "Fmeasure_score": [resultsVals["f1"]], "hamming_loss": [resultsVals["Hammingloss"]], "Accuracy_Score": [accuracyScore], "Train/Test_Size" : [testSize], "Project": [proj_name]}
metricsdf = pd.DataFrame.from_dict(dataDict)
metricsdf.to_csv(proj_name + hypothesis +  "_bertMetrics.csv", index=False)
print("Model saved")











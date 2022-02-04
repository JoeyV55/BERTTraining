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

#threshold:float = 0.5 # Best keep it in [0.0, 1.0] range
#labels = ["Util", "NLP", "APM", "Network", "DB", "Interpreter", "Error Handling", "Logging", "Lang", 
#    "Data Structure", "DevOps", "i18n", "Setup", "Logic", "Microservices", "ML", "Test", "Search",
#    "IO", "UI", "Parser", "Security", "Cloud", "Big Data", "Event Handling", "App", "GIS"]

experimentList = ["someClean_NoSMOTE","someClean_SMOTE","UNCLEAN","UNCLEAN_SMOTE", "newClean_NoSMOTE", "newClean_SMOTE"]
hypothesis = "H3"
filteringType = "UNFILTERED"
proj_title = "powertoys"
runCount = 3
validatePerEpoch = False
shouldMakeTrainTest = True
split_percent = .7
experiment = experimentList[3]
datapath =  "/scratch/jrv233/BERTWORK/" + filteringType + "/" + experiment
projectFolder = proj_title + "Bert/"
proj_name = proj_title
UNDEFINED_VAL = -9999

#Check if its a second version project (i.e. the original projects that were expanded)
if proj_title in ["jabRef", "guava", "rxjava", "mockito"]:
    proj_name += "_2"

metricspath = datapath + "/" + projectFolder 

if hypothesis == "Baseline":
    config = "issueTitle"
elif hypothesis == "H1":
    config = "issueBody"
elif hypothesis == "H2":
    config = "issueTitle + issueBody"
elif hypothesis == "H3":
    config = "issueTitle + issueBody + prComments"

print("Running project: ", proj_name, " with Hypothesis: ", hypothesis, "( ", config, " ) With configuration: ", experiment)


threshold = 0.5

labels = []

def choose_labels():
    labels = []
    if proj_name == "jabRef_2":
        #Unclean_SMOTE jabref2?
        """labels = ["Util",
        "NLP",
        "APM",
        "Network",
        "DB",
        "Interpreter",
        "Logging",
        "i18n",
        "DevOps",
        "Logic",
        "Test",
        "IO",
        "UI",
        "Security",
        "App"]"""
        #Nofilter: someClean_NOSMOTE, UNCLEAN
        #Filter: 
        """labels = ["Util",
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
        #someclean_SMOTE, unclean_SMOTE
        labels = ["Util","NLP","APM","Network","DB","Interpreter","Logging","DataStructure","i18n","DevOps",
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
    print("Labels selected: ", labels)
    return labels

def prec_rec_fscore_support_wrapper(y_pred:Tensor, y_true:Tensor, sigmoid:bool = True, thresh:float = threshold, average = 'micro', sample_weight = None, **kwargs):
    if sigmoid: y_pred = y_pred.sigmoid()
    y_pred = (y_pred > thresh).float()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    return precision_recall_fscore_support(y_true, y_pred, average=average)

def F1_micro(y_pred:Tensor, y_true:Tensor, sigmoid:bool = True, thresh:float = threshold, average = 'micro', sample_weight = None, **kwargs):
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

def findPrecisionByLabel(confMatrixList, validLabels):
    precisionDict = {}
    index = 0
    validIndicies = []
    print("ConfList: ", confMatrixList)
    originalLabels = copy.deepcopy(validLabels)
    for label in originalLabels:
        print("Prec curr label: ", label)
        print("prec index", index, " valid labels:",validLabels)
        print("prec index", index, " original labels:",originalLabels)
        print("prec index", index, " valid indicies:",validIndicies)
        currMatrix = confMatrixList[index]
        print("CurrMatrix[0]: ", currMatrix[0])
        TP = currMatrix[1][1]
        FP = currMatrix[0][1]
        precision = findPrecision(TP,FP)
        if precision == UNDEFINED_VAL:
            print("UNDEFINED PRECISION FOR LABEL: ", label)
            validLabels = removeLabel(validLabels, label)
        else:
            precisionDict[label] = precision
            validIndicies.append(index)
        index += 1

    print("PrecDict", precisionDict)
    return confMatrixList, precisionDict, validLabels, validIndicies

def removeLabel(labelList, targetLabel):
    index = 0
    for label in labelList:
        if label == targetLabel:
            del labelList[index]
            print("RemoveLabel deleted label: ", label)
            return labelList
        index+=1
    print("Couldnt find label: ", targetLabel, " in list: ", labelList)
    return labelList


def findRecallByLabel(confMatrixList, validLabels, validIndicies):
    recallDict = {}
    index = 0
    print("ValidIndicies:", validIndicies)
    print("ValidLabels:", validLabels)
    for label in validLabels:
        print("REcall index:", index)
        validIndex = validIndicies[index]
        currMatrix = confMatrixList[validIndex]
        print("Recall validInd:", validIndex)
        print("Recall currMatrix[0]: ", currMatrix[0])
        TP = currMatrix[1][1]
        FN = currMatrix[1][0]
        recall = findRecall(TP,FN)
        if not recall == UNDEFINED_VAL:
            recallDict[label] = recall    
        index += 1
    return confMatrixList, recallDict, validLabels

def findFmeasureByLabel(precisionDict, recallDict):
     print("Finding fmeasure with precisionDict: ", precisionDict, "and recall dict: ", recallDict)
     labels = list(precisionDict.keys())
     fmeasureDict = {}
     for label in labels:
         fmeasureDict[label] = findFmeasureFromPrecRec(precisionDict[label], recallDict[label])
     return fmeasureDict
     
def findFmeasureFromPrecRec(prec, rec):
    return (2*prec*rec)/(prec + rec)

def findFmeasure(TP, FP, FN):
    return TP/(TP + .5*(FP+FN))

def findPrecision(TP, FP):
    #print("Finding precision: TP: ", TP, " FP: ", FP, " TP/(TP+FP) = ", TP/(TP+FP))
    if(TP + FP) == 0:
        return UNDEFINED_VAL
    return TP/(TP+FP)

def findRecall(TP, FN):
    #print("Finding recall: TP: ", TP, " FP: ", FN, " TP/(TP + FN) = ", TP/(TP+FN))
    if(TP + FN) == 0:
        return UNDEFINED_VAL
    return TP/(TP+FN)

def computeFMetrics(inputMatrixList, validLabels):
    #First attempt to find precision, removing labels that cannot be defined
    print("VALID LABELS BEFORE PRECISION", validLabels)
    inputMatrixList, precisionDict, validLabels, validIndicies = findPrecisionByLabel(inputMatrixList, validLabels) 
    print("PRECISION DICT: ", precisionDict)
    print("VALID LABELS AFTER PRECISION: ", validLabels)
    #Then attempt to find recall, removing labels that cannot be defined again
    inputMatrixList, recallDict, validLabels  = findRecallByLabel(inputMatrixList, validLabels, validIndicies)
    print("RECALL DICT: ", recallDict)
    print("VALID LABELS AFTER RECALL: ", validLabels)
    #Then see if any more labels were removed. If so, remove them from the precall dictionary as well.
    precisionDict = removeInvalidPrecisionLabels(precisionDict, recallDict)
    print("PRECISION DICT AFTER PRECISION REMOVAL: ", precisionDict)
    print("VALID LABELS AFTER PRECISION REMOVAL: ", validLabels)

    #Finally compute fmeasure with both dictionaries, by label.
    fmeasureDict = findFmeasureByLabel(precisionDict, recallDict)
    print("FMEASURE DICT: ", fmeasureDict)
    #Compute the averages of each value
    avgPrecision, avgRecall, avgFmeasure = findFmetricAverages(precisionDict, recallDict, fmeasureDict)
    print("AvgPrecision: ", avgPrecision)
    print("AvgRecall: ", avgRecall)
    print("AvgdFmeasure: ", avgFmeasure)
    finalFmeasure = findFmeasureFromPrecRec(avgPrecision, avgRecall)
    print("FMEASURE FROM FINAL PRECISION AND RECALL: ", finalFmeasure)
    return avgPrecision, avgRecall, finalFmeasure

def removeInvalidPrecisionLabels(precisionDict, recallDict):
    precKeys = copy.deepcopy(list(precisionDict.keys()))
    for label in precKeys:
        if label not in recallDict:
            del precisionDict[label]
    return precisionDict 

def findFmetricAverages(precisionDict, recallDict, fmeasureDict):
    labels = list(precisionDict.keys())
    sumPrecision = 0.0
    sumRecall = 0.0
    sumFmeasure = 0.0
    avgPrecision = 0.0
    avgRecall = 0.0
    avgFmeasure = 0.0
    labelCount = len(labels)
    for label in labels:
        sumPrecision += precisionDict[label]
        sumRecall += recallDict[label]
        sumFmeasure += fmeasureDict[label]
    avgPrecision = sumPrecision/labelCount
    avgRecall = sumRecall/labelCount
    avgFmeasure = sumFmeasure/labelCount
    return avgPrecision, avgRecall, avgFmeasure

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
        if(metricDict[label] == 2.0):
            continue
        count+=1
        total += metricDict[label]
    return total / count

"""
Used to find the total True Positives, False Positives, and False Negatives for use in computing the micro fmeasure.

    'micro':
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
"""
def findTPFNFP(confMatrixList):
    totalTP = 0
    totalFP = 0
    totalFN = 0
    for matrix in confMatrixList:
        print("First matrix: ", matrix)





print("Started fast-bert, emptying cache")
torch.cuda.empty_cache()


DATA_PATH = './data/'
LABEL_PATH = './data/'
OUTPUT_PATH = './output/'

labels = choose_labels()
if len(labels) == 0:
    sys.exit("No labels found for projectname inputted: " + proj_name)

testSize = split_percent


print("Building Box")
currRun = 0
while currRun < runCount:
    print('===============\n\nCURR RUN\n\n====================', currRun)
    currRun += 1

    #Build train and test csvs if desired.
    if shouldMakeTrainTest:
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
        #del train["pr"]
        #del test["pr"]
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
    metrics = [{'name': 'accuracy', 'function': accuracy}, {'name': 'f1', 'function': F1}, {'name': 'Hammingloss', 'function': Hamming_loss_mod},
    {'name' : 'prec_rec_fmeas_support', 'function': prec_rec_fscore_support_wrapper}, {'name': 'confmatrix', 'function': confusion_matrix_by_label},{'name': 'f1bylabel', 'function': F1_by_label},{'name': 'accuracybylabel', 'function': accuracy_by_label}]

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
    learner.fit(epochs=11,
                lr=6e-5,
                validate=validatePerEpoch, 	# Evaluate the model after each epoch
                return_results=True,
                schedule_type="warmup_cosine",
                optimizer_type="lamb")

    print("Lr fit finished")

    #Get the return with the results of this prediction
    resultsVals = learner.validate()

    print(resultsVals)


    #Get array for confusion matrix
    confMatrixList = resultsVals["confmatrix"]

    #recallDict = findRecallByLabel(confMatrixList)
    ##precisionDict = findPrecisionByLabel(confMatrixList)

    accuracyDict = resultsVals["accuracybylabel"]
    ##hamLossDict = findHammingLossByLabel(accuracyDict)

    #recallAvg = sanityCheck(recallDict)
    #precisionAvg = sanityCheck(precisionDict)

    accuracyScore = 1-resultsVals["Hammingloss"]


    print("\n==================\n\nMetrics from prec, rec, fmeas, support\n============================")
    #avgPrecVal, avgRecallVal, avgFMeasureVal = computeFMetrics(confMatrixList, copy.deepcopy(labels))
    metricTuple = resultsVals["prec_rec_fmeas_support"]
    microPrec = metricTuple[0]
    microRec = metricTuple[1]
    microFmeas = metricTuple[2]
    


    print("\n==================\n\nSanity Check Averages (0s included)\n============================")
    #print("Hamming loss sanity check average: ", sanityCheck(hamLossDict))
    #print("Recall sanity check average: ", sanityCheck(recallDict))
    #print("Precision loss sanity check average: ", sanityCheck(precisionDict))


    print("\n==================\n\nNonZero averages\n============================")
    #print("Recall skipping 0s", averageNonZeros(recallDict))
    #print("Precision skipping 0s", averageNonZeros(precisionDict))

    learner.save_model()

    dataDict = {"Hypothesis" : [hypothesis], "Precision" : microPrec, "Recall": microRec, "Fmeasure_score": microFmeas, "hamming_loss": [resultsVals["Hammingloss"]], "Accuracy_Score": [accuracyScore], "Train/Test_Size" : [testSize], "Project": [proj_name]}

    metricsdf = pd.DataFrame.from_dict(dataDict)
    metricsFile = metricspath + proj_name + "_" + hypothesis + "_" + filteringType + "_bertMetrics.csv"
    if not os.path.exists(metricsFile):
        metricsdf.to_csv(metricsFile, index=False)
    else:
        with open(metricsFile, 'a', newline='') as metricscsv:
            fieldNames = ["Hypothesis","Precision","Recall","Fmeasure_score","hamming_loss","Accuracy_Score","Train/Test_Size","Project"]
            newRow = {"Hypothesis" : hypothesis, "Precision" : microPrec, "Recall": microRec, "Fmeasure_score": microFmeas, "hamming_loss": resultsVals["Hammingloss"], "Accuracy_Score": accuracyScore, "Train/Test_Size" : testSize, "Project": proj_name}

            #rowFields = [hypothesis, -9999,-9999, resultsVals["f1"], resultsVals["Hammingloss"], accuracyScore, testSize]
            writer = csv.DictWriter(metricscsv, fieldnames=fieldNames)
            writer.writerow(newRow)

sys.stdout = open(metricspath + "resultsVals.txt", 'a')
#Write all the metrics to a file in metricsPath

if runCount > 0:
    print("Run: ", runCount, resultsVals)

    metricsDf = pd.read_csv(metricsFile)
    print(metricsDf.head())
print("Mean vals: ", metricsDf.mean(axis=0))
print("Model saved")
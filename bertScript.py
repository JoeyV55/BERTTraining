#!/usr/bin/env python
# coding: utf-8
from bertUtils import *
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
from sklearn.model_selection import train_test_split, ShuffleSplit

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
hypothesis = "H1"
config = "H11"
filteringType = "UNFILTERED"
proj_title = "jabRef"
runCount = 10
validatePerEpoch = False
crossValidating = True


experiment = experimentList[3]
datapath =  "/scratch/jrv233/BERTWORK/" + filteringType + "/" + experiment
projectFolder = proj_title + "Bert/"
proj_name = proj_title
social_metrics = True
#70% train, 30% test default, change to swap to other configs.
UNDEFINED_VAL = -9999

#Check if its a second version project (i.e. the original projects that were expanded)
if proj_title in ["jabRef", "guava", "rxjava", "mockito","all"]:
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



"""
    Experiment methods (if needed)
"""
#TODO: Validate after logic and see if metrics are truely impossible to extract from fast bert
"""def runH10():
    print("\n=====================\nRUNNING H10 TEST ON POWERTOYS AFTER TRAINING WITH COMBINED MODEL\n================================")
    texts = ["fix editor cli parsing if comma is decimal sep in locale <!-- enter a brief description/summary of your pr here what does it fix/what does it change/how was it tested (even manually if necessary)? -->fix fancyzoneseditor cli parsing of dpi on locales with comma as decimal separator<!-- other than the issue solved is this relevant to any other issues/existing prs? --> related existing issue #449<!-- provide a more detailed description of the pr other things fixed or any additional comments/features here -->although the cli should be constructed with c locale (c/c++ invariant in c#) by the fancyzones application i wasnt completely certain about that so i resorted to using a fallback mechanism of trying both invariant and current cultures for float parsing<!-- describe how you validated the behavior add automated tests wherever possible but list manual validation steps taken as well -->## validation steps performedtested manually by setting computer language eg to finnish (comma as decimal separator) and regression tested with en-us as computer culture as well as finnish with forced period as decimal separator hi @mickutthank you for fixing this!",
    "fixes for zone resize <!-- enter a brief description/summary of your pr here what does it fix/what does it change/how was it tested (even manually if necessary)? -->## summary of the pull requestresizing zones was not working in some cases<!-- please review the items on the pr checklist before submitting-->## pr checklist* [x] closes #257 #409 #427 <!-- provide a more detailed description of the pr other things fixed or any additional comments/features here -->## detailed description of the pull request / additional commentsthe logic used by `move()` works only for dragging the entire zone but it fails when dragging a top/left edge if the right/bottom edge is on the working area edgeit was also causing the zone to drift right/bottom when resized to its minimum width/heighti split the methods to simplify the logic: - `move()` is used only to move the zone - `sizemove()` (fell free to suggest a better name) is used when dragging the left/top edge - `size()` is used when dragging the right/bottom edge also set `c_minzonewidth` to 64 and `c_minzoneheight` to 72 to reflect the actual minimum dimensions<!-- describe how you validated the behavior add automated tests wherever possible but list manual validation steps taken as well -->## validation steps performedmanual test"]
    res = learner.predict_batch(texts)
    print("type of res: ", type(res))
    print(res)"""

print("Started fast-bert, emptying cache")
torch.cuda.empty_cache()


DATA_PATH = './data/'
LABEL_PATH = './data/'
OUTPUT_PATH = './output/'

labels = choose_labels(proj_name)
if len(labels) == 0:
    sys.exit("No labels found for projectname inputted: " + proj_name)



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


def findHammingLossByLabel(accuracyDict):
    hammingLossDict = {}
    for label in labels:
        hammingLossDict[label] = 1-accuracyDict[label]
    return hammingLossDict

split_percent = .7
size_test = 1.0 - split_percent
while(split_percent < .85):
    print("=========================\nSPLITTING WITH SIZE_TEST: ", size_test, "\n====================")
    split_percent_display = str(split_percent * 100) + "%"
    print("Building Box")
    currRun = 0
    #Read whole binary to csv.
    dataset_smote = pd.read_csv('./data/binaryNew.csv')
    print("dataset_smote COLUMNS: ", dataset_smote.columns)
    X = dataset_smote['corpus']

    #SHUFFLE SPLIT SECTION
    #Run the shuffle split and init vars for it
    train = []
    test = []

    splits = 10

    rs = ShuffleSplit(n_splits = splits, test_size = size_test, random_state=52)
    #print("\n============================\nShuffleSPLITS\n=========================================\n")
    #print(rs.get_n_splits(X))

    for train_index, test_index in rs.split(X):
        #print("%s %s" % (train_index, test_index))
        train.append(train_index)
        test.append(test_index)    
        #print("Train 0:", train[0])

    train_test_splits = []

    for i in range(0, len(train)):
        train_test_pair = [0,0]

        num_columns = len(dataset_smote.columns)

        data = dataset_smote.iloc[train[i]]



        #data.to_csv(classifierFeatureInput+'train'+str(i)+'.csv', encoding='utf-8', sep=',')
        #JabRef has no prNumber column so it must select from col 0. TODO fix this
        train_data = data.iloc[:,1:num_columns] # all
        
        #print("train_data: \n", train_data.head())

        data = dataset_smote.iloc[test[i]]

        #data.to_csv(classifierFeatureInput+'test'+str(i)+'.csv', encoding='utf-8', sep=',')
        test_data = data.iloc[:,1:num_columns]
        #print("test_data: \n", test_data.head())
        
        train_test_pair[0] = train_data
        train_test_pair[1] = test_data

        train_test_splits.append(train_test_pair)

    #BEGIN RUNNING BERT
    currRun = 0
    while currRun < runCount:
        print('===============\n\nCURR RUN\n\n====================', currRun)
        if crossValidating:
            curr_split = train_test_splits[currRun]
            curr_train = curr_split[0]
            curr_test = curr_split[1]
            #Check to see if we are overwriting the files.
            curr_train.to_csv('data/binaryTrain.csv', index=False) 
            curr_test.to_csv('data/binaryTest.csv', index=False)
            print("CURR TRAIN ", currRun, curr_train.head())
            print("CURR TEST ", currRun, curr_test.head()) 
        currRun += 1
        

        #Build train and test csvs if desired.
        """if shouldMakeTrainTest:
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
            test.to_csv('./data/binaryTest.csv', index=False)"""
        


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

        #THIS IS WRONG AND NEEDS TO COME FROM SKLEARN FOR MULTILABEL!
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

        dataDict = {"Hypothesis" : [hypothesis], "Precision" : microPrec, "Recall": microRec, "Fmeasure_score": microFmeas, "hamming_loss": [resultsVals["Hammingloss"]], "Accuracy_Score": [accuracyScore], "Train/Test_Size" : [size_test], "Project": [proj_name]}

        metricsdf = pd.DataFrame.from_dict(dataDict)
        metricsFile = metricspath + proj_name + "_" + hypothesis + "_" + filteringType + str(split_percent_display) + "_SocialMetrics" + str(social_metrics) + "_bertMetrics.csv"
        if not os.path.exists(metricsFile):
            print("NO metrics exist, writing new file at path", metricsFile)
            metricsdf.to_csv(metricsFile, index=False)
        else:
            print("Metrics exist, appending file at path", metricsFile)
            with open(metricsFile, 'a', newline='') as metricscsv:
                fieldNames = ["Hypothesis","Precision","Recall","Fmeasure_score","hamming_loss","Accuracy_Score","Train/Test_Size","Project"]
                newRow = {"Hypothesis" : hypothesis, "Precision" : microPrec, "Recall": microRec, "Fmeasure_score": microFmeas, "hamming_loss": resultsVals["Hammingloss"], "Accuracy_Score": accuracyScore, "Train/Test_Size" : size_test, "Project": proj_name}

                #rowFields = [hypothesis, -9999,-9999, resultsVals["f1"], resultsVals["Hammingloss"], accuracyScore, testSize]
                writer = csv.DictWriter(metricscsv, fieldnames=fieldNames)
                writer.writerow(newRow)

    
    #Write all the metrics to a file in metricsPath
    if runCount > 0:
        print("Run: ", runCount, resultsVals)
        sys.stdout = open(metricspath + hypothesis + "resultsVals.txt", 'a')
        metricsDf = pd.read_csv(metricsFile)
        print(metricsDf.head())
        print("Mean vals: ", metricsDf.mean(axis=0))
        print("Model saved")
        sys.stdout = sys.__stdout__

    split_percent += .1
    size_test = 1 - split_percent

#if proj_name in ["powertoys_rmca", "rmca","powertoys"]:
 #   runH10()




#!/usr/bin/env python
# coding: utf-8


#labels = ['toxic','severe_toxic','obscene','threat','insult','identity_hate']
DATA_PATH = './data/'
LABEL_PATH = './data/'
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

from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging
from transformers import BertTokenizer
from pathlib import Path
import torch

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
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc

print("Started fast-bert, emptying cache")
torch.cuda.empty_cache()

labels = ["Util", "NLP", "APM", "Network", "DB", "Interpreter", "Error Handling", "Logging", "Lang", 
    "Data Structure", "DevOps", "i18n", "Setup", "Logic", "Microservices", "ML", "Test", "Search",
    "IO", "UI", "Parser", "Security", "Cloud", "Big Data", "Event Handling", "App", "GIS"]

DATA_PATH = './data/'
LABEL_PATH = './data/'
OUTPUT_PATH = './output/'

print("Building Box")

args = Box({
    "run_text": "presto multilabel",
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
                          #train_file='train.csv',
                          #val_file='val.csv',
                          #label_file='labels.csv',
                          train_file='binaryNew.csv',
                          val_file='binaryNew.csv',
                          label_file='labelsNew.csv',
                          #text_col='text',
                          text_col='corpus',
                          label_col=labels,
                          batch_size_per_gpu=16,
                          max_seq_length=512,
                          multi_gpu=False,
                          multi_label=True,
                          model_type='bert')


# In[15]:


from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging

logger = logging.getLogger()

device_cuda = torch.device("cuda")
print("Got cuda device")
metrics = [{'name': 'accuracy', 'function': accuracy}]

OUTPUT_DIR = './output/'

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='bert-base-uncased',
						metrics=metrics,
						device=device_cuda,
                        #device=None,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=False,
						is_fp16=True,
						multi_label=True,
						logging_steps=50)


# In[ ]:

print("Starting lr find")
#learner.lr_find(start_lr=1e-5,optimizer_type='lamb')
print("Finished lr find")

# In[33]:

print("Starting fit")
learner.fit(epochs=6,
			lr=6e-5,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="lamb")

#print("Finished fit")
#learner.save_model()
sys.exit("Saved ZEEMODEL")
#print("Saved model")

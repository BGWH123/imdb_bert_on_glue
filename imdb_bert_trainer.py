# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import sys#提供自我访问接口
import logging
import datasets#计成一个高效、易用、可扩展的对象，可以方便地进行处理和操作
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification, DataCollatorWithPadding
#这是一个用于处理文本的分词器（tokenizer）
#这是一个用于序列分类任务的 BERT 模型。序列分类任务是指将整个文本序列分类到一个或多个类别中
#这是一个数据整理器（collator），用于将多个样本组合成一个批次（batch）以供模型训练使用
from transformers import Trainer,TrainingArguments
#TrainingArguments 是一个配置类，用于定义训练过程中的各种参数和选项。
#Trainer它接受一个模型、数据集、训练参数等，并提供了一个简单的接口来启动训练过程。
from sklearn.model_selection import train_test_split

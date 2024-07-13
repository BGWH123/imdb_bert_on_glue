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
train=pd.read_csv("/kaggle/input/cola-data/CoLA/train.tsv", header=0, delimiter="\t", quoting=3)
test=pd.read_csv("/kaggle/input/cola-data/CoLA/test.tsv", header=0, delimiter="\t", quoting=3)
temp=pd.read_csv("/kaggle/input/labeledtraindata/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
#设置日志
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info(r"running %s" % ''.join(sys.argv))
train,val=train_test_split(train,test_size=.2)#将数据集分割成训练集和测试集 8:2
#设置字典存储数据
train_dict= {'label': train.iloc[:,1], 'text': train.iloc[:,3]}
val_dict={'label': val.iloc[:,1], 'text': val.iloc[:,3]}
test_dict={'text':test.iloc[:,1]}
train_dataset=datasets.Dataset.from_dict(train_dict)
val_dataset = datasets.Dataset.from_dict(val_dict)
test_dataset = datasets.Dataset.from_dict(test_dict)
tokenizer=BertTokenizerFast.from_pretrained('bert-base-uncased')#预训练一个分词器，将其文本转化成可以理解的格式
def preprocess_function(examples):
    return tokenizer(examples['text'],truncation=True)
#分词处理文本数据 确保长度适合输入    
#将训练数据集的文本转换为模型可理解的格式
tokenized_train=train_dataset.map(preprocess_function,batched=True)#，即一次处理多个样本
tokenized_val = val_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)
#填充长度
data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
#加载模型
model=BertForSequenceClassification.from_pretrained("bert-base-uncased")
#准确率指标
metric=datasets.load_metric("accuracy")
def compute_metrics(eval_pred):
    logits,labels=eval_pred
    predictions=np.argmax(logits,axis=-1)
    return metric.compute(predictions=predictions,references=labels)
#初始化
training_args = TrainingArguments(
    output_dir='./checkpoint',  # 输出目录，用于存放训练过程中的检查点文件
    num_train_epochs=3,  # 总的训练轮数
    per_device_train_batch_size=16,  # 训练时每个设备上的批处理大小
    per_device_eval_batch_size=32,  # 评估时每个设备的批处理大小
    warmup_steps=500,  # 学习率调度器的预热步数
    weight_decay=0.01,  # 权重衰减的强度，用于正则化
    logging_dir='./logs',  # 训练日志的存储目录
    logging_steps=100,  # 日志记录的步数间隔
    save_strategy="no",  # 不保存模型检查点
    evaluation_strategy="epoch"  # 评估策略，每个epoch结束后进行评估
)
trainer = Trainer(
    model=model,  # 初始化的 Transformers模型，用于训练
    args=training_args,  # 训练参数，如上文定义
    train_dataset=tokenized_train,  # 训练数据集，已经过预处理和分词
    eval_dataset=tokenized_val,  # 评估数据集，用于模型性能评估
    tokenizer=tokenizer,  # 分词器，用于数据的分词处理
    data_collator=data_collator,  # 数据整理器，用于批处理数据的填充等操作
    compute_metrics=compute_metrics,  # 计算评价指标的函数，用于评估模型性能
)
#训练
!wandb off
trainer.train()
prediction_outputs=trainer.predict(tokenized_test)
test_pred=np.argmax(prediction_outputs[0],axis=-1).flatten()#续输出（logits）转换为按列离散的类别预测，
 #即将每个样本的得分最高的类别索引提取出来按列，并展平成一维数组 
print(test_pred)
result_output = pd.DataFrame(data={"id":test["index"],"sentiment":test_pred})
result_output.to_csv("/kaggle/working/checkpoint/bert_trainer.csv",index=False,quoting=3)
logging.info('result saved!')

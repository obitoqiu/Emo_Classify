# coding:utf-8
import numpy as np
from cal import tf, idf, tf_idf
from collections import Counter
import re

np.set_printoptions(threshold=np.NaN)  #输出不限列宽

# 初始化特征表
# word用词典{'word':编号}
# label用列表  有四类label：angry sad happy others
file = open('emoji/all.txt', 'r', encoding='utf-8')
line = file.readline()
emotioncount = []
wordcount = {}
index = 0
while line:
    line = line[0:line.find(' ')]
    wordcount[line] = index
    index = index + 1
    line = file.readline()
file.close()
# print(wordcount)  13156


# make train matrix 前27000条
file = open('starterkitdata/train.txt', 'r', encoding='utf-8')
line = file.readline()
trainsum = 27000
traincount = []
index = 0
while line and index <= trainsum:
    emotioncount.append(line[line.rfind('\t')+1:line.rfind('\n')])
    line = line[line.find('	')+1: line.rfind('	')]
    line = line.replace('\t', ' ')
    traincount.append(line)
    index = index + 1
    # print(line)
    line = file.readline()
file.close()

# 创建一个全0矩阵，使得行数(即把所有词视为特征)等于词数，列数等于训练集条数
trainmatrix = np.zeros(shape=(wordcount.__len__(), traincount.__len__()))
id = 0
for line in traincount:
    for word in line.split():
        if word in wordcount:
            dem = wordcount[word]
            trainmatrix[dem, id] = trainmatrix[dem, id] + line.count(word)
    id = id + 1
    # print(id)


emomatrix = np.array(emotioncount).T              # 把列表emotioncount数组化后做一个转置，则行为训练集条号，每行1列为label
weights_train = tf_idf(trainmatrix)               # 对文档共现矩阵做tf-idf处理得到文档向量

np.save('trainvec.npy', weights_train.T)          # 保存转置后的文档向量,转置后行为训练集条数，列为特征数
np.save('trainemo.npy', emomatrix)                # 保存

emotioncount.clear()

# make predict matrix 后3000条  即验证集  操作步骤同上
file = open('starterkitdata/train.txt', 'r', encoding='utf-8')
line = file.readline()
start = 27001
predictcount = []
index = 0
while line:
    if index >= start:
        emotioncount.append(line[line.rfind('\t')+1:line.rfind('\n')])
        line = line[line.find('	')+1: line.rfind('	')]
        line = line.replace('\t', ' ')
        predictcount.append(line)

        # print(line)
    else:
        pass
    line = file.readline()
    index = index + 1
file.close()

# 创建一个全0矩阵，使得行数等于词数，列数等于预测集条数
predictmatrix = np.zeros(shape=(wordcount.__len__(), predictcount.__len__()))
id = 0
for line in predictcount:
    for word in line.split():
        if word in wordcount:
            dem = wordcount[word]
            predictmatrix[dem, id] = predictmatrix[dem, id] + line.count(word)
    id = id + 1
    # print(id)
emomatrix = np.array(emotioncount).T
weights_pre = tf_idf(predictmatrix)

np.save('predictvec.npy', weights_pre.T)
np.save('predictemo.npy', emomatrix)







'''
#把矩阵保存为文本向量  形如 label demession1：value1 demmession2：value....
with open('E:/semeval2019/starterkitdata/trainvec.txt', 'w') as f:
    file = open('E:/semeval2019/starterkitdata/train.txt', 'r', encoding='utf-8')
    line = file.readline()
    id = 0
    while line and id<= trainsum:
        string = line[line.find('	')+1:line.rfind('\n')]
        f.write(string)
        f.write(str(trainmatrix[:, id]).replace('[', ' ').replace(']', '\n'), end='')
        id = id + 1
        print(id)
        line = file.readline()

    file.close()
'''


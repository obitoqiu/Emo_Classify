import numpy as np
import math

def tf(matrix):
    sm = np.sum(matrix, axis=0)+1e-20    # 计算每个文档的总词数
    #print(sm)
    return matrix / sm    # 每个词的词频除以每个文档的词频

def idf(matrix):

    D = matrix.shape[1]  # 文档总数
    #print(D)

    j = np.sum(matrix > 0, axis=1)+1e-20    # 包含每个词的文档数
    #print(j)
    return np.log(D/j)

def tf_idf(matrix):
    return tf(matrix) * idf(matrix).reshape(matrix.shape[0], 1)

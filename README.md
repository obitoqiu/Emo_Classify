# Emo_Classify
## use SVM in sklearn to achieve text emotion classifying
### 环境
- **Environments:Python 3.5**
- **Libs:sklearn 0.0,numpy 1.15.4**
### 代码实现
`cal.py`: 对文档词汇共现矩阵进行tf-idf求文档内容向量</br>
`docvec.py`: 读入训练集构造相关特征表并将文档向量以.npy的数组格式保存</br>
`SVC.py`: 采用SVM进行训练和预测</br>
`train.txt`: 数据集  内容为 id Turn1 Turn2 Turn3 Lable</br>
`all.txt` : 统计得到的单词、表情字典</br>
运行顺序：`docvec.py`->`SVC.py`
###  实验结果
**在四标签的约45000条数据中按训练集：验证集=3:1，进行训练并测试，召回率为49.7%**

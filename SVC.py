import pickle
import numpy as np
from sklearn.svm import SVC
'''读取svm中保存的向量集及label集'''
trainvec = np.load('trainvec.npy')
trainemo = np.load('trainemo.npy')
predictvec = np.load('predictvec.npy')
predictemo = np.load('predictemo.npy')


# 用Dataprot()、Dataprop()两个类的方法分别get data
class Dataprot():
    def getY(self):
        self.y_data = trainemo[:, ]
        return self.y_data

    def getX(self):
        self.x_data = trainvec[:, :]
        return self.x_data


class Dataprop():
    def getY(self):
        self.y_data = predictemo[:, ]
        # print(self.y_data)
        return self.y_data

    def getX(self):
        self.x_data = predictvec[:, :]
        #print(self.x_data)
        return self.x_data


class SVM(object):
    '''
    进入SVM的每条训练和预测数据标准格式都是：
    条号 feature1 feature2 feature3 .... feature 13156  label
     0    x1      x2       x3              x13156        y
    '''

    def __init__(self, train_dataproc, predict_dataproc):
        self.__train_data = train_dataproc
        self.__predict_data = predict_dataproc
        self.__svc = SVC()

    # train data and general a model
    def train(self):
        train_y = self.__train_data.getY()                                  #所有训练文本类别
        train_x = self.__train_data.getX()                                  #所有训练文本内容向量
        self.__svc.fit(train_x, train_y)
        self.model_presistence()

    # predict the label
    def predict(self):
        predict_y = self.__predict_data.getY()
        predict_x = self.__predict_data.getX()
        res = self.__svc.predict(predict_x)
        accu = 0

        for i in range(len(predict_y)):
            if predict_y[i] == res[i]:
                accu = accu + 1

        accu = accu/len(predict_y)
        print('the accuracy is %f' %accu)

    # save a model in .pkl file
    def model_presistence(self):
        fileObject = open('SVM.pkl', 'wb')
        pickle.dump(self.__svc, fileObject)                                 #将SVC持久化
        fileObject.close()

    # load a model from a .pkl file
    def read_model(self):
        fileName = 'SVM.pkl'
        fileObject = open(fileName, 'rb')
        self.__svc = pickle.load(fileObject)                                #读取SVC


if __name__ == '__main__':
    train_data = Dataprot()
    predict_data = Dataprop()

    svm = SVM(train_data, predict_data)

    #开始训练
    svm.train()
    print('finished train model')

    #开始预测
    svm.predict()






import numpy as np
import random

#加载数据
def loadDataSet():
    trainMat = []; testMat = []
    trainlabelMat = []; testlabelMat = []
    fr = open('题目4个人收入预测.csv')
    t=0 # t判定当前读取的第几组数据
    for line in fr.readlines():
        currline = line.strip().split(',')
        lineArr = []
        lineArr.append(1.0) #每个样本添加一个值为1的特征，和对应的系数相乘后作为截距
        if t < 3000: #前3000组作为训练集
            for i in range(1,58): #第一列特征不载入，因为代表的含义是ID，和最终的判定结果应该无关
                lineArr.append(float(currline[i]))
            trainMat.append(lineArr) 
            trainlabelMat.append(float(currline[58]))
        else: #后1000组作为测试集
            for i in range(1,58):
                lineArr.append(float(currline[i]))
            testMat.append(lineArr)
            testlabelMat.append(float(currline[58]))
        t = t+1
    return trainMat,trainlabelMat,testMat,testlabelMat

#归一化处理
def dataprocess(trainMat):
    trainMat = np.mat(trainMat)
    m,n = trainMat.shape
    for i in range(n):
        meanval = np.mean(trainMat[:,i]) #求每种特征的均值
        stdval = np.std(trainMat[:,i]) #求每种特征得标准差
        if stdval != 0.0: #判定标准差是否为0，为0的话在进行归一化处理时0最为除数是无意义的
            trainMat[:,i] = (trainMat[:,i] - meanval)/stdval #标准差不为0则进行归一化处理
        else:
            trainMat[:,i] = 1 #标准差为0则将该特征值都设为1
    return trainMat

#sigmoid函数
def sigmoid1(inx):
    return 1.0/(1+np.exp(-inx))

#梯度下降优化算法
def graddec(trainMat,trainlabelMat):
    dataMat = np.mat(trainMat)
    labelMat = np.mat(trainlabelMat).transpose()
    m,n = np.shape(dataMat)
    alpha = 0.1 #设定学习率
    maxcycles = 200 #设定迭代次数
    weights = np.zeros((n,1)) #初始化特征系数全0
    loss = [] #损失值
    for k in range(maxcycles):
        #alpha = 8/(k+1)+0.1 #动态调整学习率
        h = sigmoid1(dataMat * weights)
        error = h - labelMat
        weights = weights - alpha /m * np.transpose(dataMat) * error #未正则化
        # weights = (1 - alpha * u/m) * weights - alpha /m * np.transpose(dataMat) * error #正则化后
        if (k+1) % 10 == 0:    
            loss.append(- float((labelMat.T * np.log(h + 0.0001) + (1 - labelMat.T) * np.log(1.0001 - h))) /m) #损失函数值
    return weights,loss

#
def stocgraddec(trainMat,trainlabelMat,numIter):
    dataMat = np.mat(trainMat)
    m,n = np.shape(dataMat)
    weights = np.zeros((n, 1))
    for i in range(numIter):
        dataIndex = list(range(m))
        loss = 0
        for j in range(m):
            alpha =  0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid1(sum(dataMat[randIndex] * weights))
            error = sum(h - trainlabelMat[randIndex])
            weights = weights - alpha /m * float(error) * np.transpose(dataMat[randIndex])
            del(dataIndex[randIndex])
            if (i + 1) % 100 ==0:
                loss += -float((trainlabelMat[randIndex] * np.log(h + 0.0001) + (1 - trainlabelMat[randIndex]) * np.log(1.0001 - h)))
        """
        if (i + 1) % 100 ==0:
             print(loss/m)
        """
    return weights 


#鉴别函数
def classify(x,weights):
    prob = sigmoid1(sum(x * weights))
    if prob > 0.5 : return 1.0 #测试样本的sigmoid函数值大于0.5则将该样本标为类别1
    else : return 0.0

# 测试
def incometest(testMat,testlabelMat,weights):
    errorcount = 0; numcount = 1000.0
    for i in range(1000):
        if classify(np.array(testMat[i]),weights) != testlabelMat[i]:
            errorcount += 1 #和预设标签不同则错误数加一
    truerate = (numcount - errorcount)/numcount #求正确率
    print(truerate)
    return truerate


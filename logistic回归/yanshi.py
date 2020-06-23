import numpy as np
import logistic
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']

trainMat,trainlabelMat,testMat,testlabelMat = logistic.loadDataSet()

trainMat = logistic.dataprocess(trainMat)
testMat = logistic.dataprocess(testMat) #进行归一化处理

weights,loss = logistic.graddec(trainMat,trainlabelMat)
truerate = logistic.incometest(testMat,testlabelMat,weights) #采用批处理的梯度下降算法

"""
xcord = list(range(10,201,10)) ; ycord = []
alpha = 0.001
while alpha <= 10:
    weights,loss = logistic.graddec(alpha,trainMat,trainlabelMat)
    truerate = logistic.incometest(testMat,testlabelMat,weights) #采用批处理的梯度下降算法
    ycord.append(loss)
    alpha *= 10
plt.plot(xcord, ycord[0], marker = '*' , ms = 2, label = 'alpha = 0.001')
plt.plot(xcord, ycord[1], marker = '*' , ms = 2, label = 'alpha = 0.01')
plt.plot(xcord, ycord[2], marker = '*' , ms = 2, label = 'alpha = 0.1')
plt.plot(xcord, ycord[3], marker = '*' , ms = 2, label = 'alpha = 1')
plt.plot(xcord, ycord[4], marker = '*' , ms = 2, label = 'alpha = 10')
plt.legend(loc = 2)
plt.xlabel('迭代次数')
plt.ylabel('损失函数值')
plt.ylim(0,1)
plt.show()


weights = logistic.stocgraddec(trainMat,trainlabelMat,500)
logistic.incometest(testMat,testlabelMat,weights) #采用随机梯度下降算法且学习率随着迭代次数的增加而逐渐下调至某一固定非0值
"""
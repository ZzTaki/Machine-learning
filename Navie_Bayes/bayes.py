'''
Created on Oct 19, 2010

@author: Peter
'''
import numpy as np
import re
import random

"""
函数说明：将切分后的实验样本整理成不重复的词条列表，即创建词汇表
输入：所有的样本数据组成的列表
返回：词汇表
"""                 
def createVocabList(dataSet):
    vocabSet = set([])  #创建一个空的集和
    for document in dataSet:
        vocabSet = vocabSet | set(document) #取并集
    return list(vocabSet)

"""
函数说明：按照词集模型规则，根据词汇表，将输入的邮件向量化，向量的每个元素为1或0
输入：词汇表和切分后的词条列表
返回：词集模型的文档向量
"""
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList) #创建一个全是0的向量
    for word in inputSet: #遍历每个词条
        if word in vocabList: #如果该词在词汇表中，则置为1
            returnVec[vocabList.index(word)] = 1 
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec #返回文档向量

"""
函数说明：朴素贝叶斯分类器的训练函数
输入：训练文档矩阵和训练类别标签向量
返回：正常邮件、垃圾邮件类的各词出现的条件概率以及文档属于垃圾邮件的概率
"""
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix) #计算训练的文档数
    numWords = len(trainMatrix[0]) #计算每篇文档的词条数
    pAbusive = sum(trainCategory)/float(numTrainDocs) #文档属于垃圾邮件的概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords) #拉普拉斯平滑 
    p0vnumtotal = 2.0; p1vnumtotal = 2.0        #拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1: 
            p1Num += trainMatrix[i] #统计垃圾邮件中每个词出现的次数
            p1vnumtotal += sum(trainMatrix[i]) #统计垃圾邮件的总词数
        else:
            p0Num += trainMatrix[i] #统计正常邮件中每个词出现的次数
            p0vnumtotal += sum(trainMatrix[i]) #统计正常邮件的总词数
    p1Vect = np.log(p1Num/p1vnumtotal)         
    p0Vect = np.log(p0Num/p0vnumtotal)          #取对数，防止下溢出
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

"""
函数说明：按照词袋模型规则，根据词汇表，将输入的词条列表向量化，向量的每个元素为1或0
输入：词汇表和切分后的词条列表
返回：词集模型的文档向量
"""         
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList) #创建一个全为0的向量
    for word in inputSet: #遍历每个词条
        if word in vocabList: #该词条存在于词汇表中，则计数加1
            returnVec[vocabList.index(word)] += 1
    return returnVec

"""
函数说明：接收大字符串并将其按照某种规则解析为字符串类别
"""
def textParse(bigString):   #将字符串转化为字符串列表
    listOfTokens = re.split(r'\W+', bigString) #将非字符、非数字作为切分标志进行字符串切分
    return [voca.lower() for voca in listOfTokens if len(voca) > 2] #除去长度小于3的字符串，并将剩余字符串转化为小写
    
"""
函数说明：测试朴素贝叶斯分类器，使用朴素贝叶斯进行交叉验证
"""
def spamTest():
    docList=[]; classList = []; fullText =[]
    for i in range(1,26): #遍历25个垃圾邮件和正常邮件
        wordList = textParse(open('email/spam/%d.txt' % i,'r').read()) #读取每个垃圾邮件，并将字符串转化为字符串列表
        docList.append(wordList) 
        fullText.extend(wordList)
        classList.append(1) #垃圾邮件标为1
        wordList = textParse(open('email/ham/%d.txt' % i,'r').read()) #读取每个正常邮件，并将字符串转化为字符串列表
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0) #正常邮件标为0
    vocabList = createVocabList(docList) #创建词汇表
    trainingSet = list(range(50)); testSet=[]           #创建存储训练集索引值的列表和测试集索引值的列表
    for i in range(10): #从50个邮件中，随机挑选40个作为训练集，10个作为测试集
        randIndex = int(random.uniform(0,len(trainingSet))) #随机选取索引值
        testSet.append(trainingSet[randIndex]) #添加测试集的索引值
        del(trainingSet[randIndex])  #删除训练集列表中添加到测试集的索引值
    trainMat=[]; trainClasses = [] #创建训练集矩阵和训练集类别标签向量
    for docIndex in trainingSet: #遍历训练集
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex])) #将生成的词袋模型添加到训练矩阵中
        trainClasses.append(classList[docIndex]) #将类别添加到训练集类别标签向量中
    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses)) #训练朴素贝叶斯模型
    errorCount = 0 #错误分类计数
    for docIndex in testSet:        #遍历测试集
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex]) #测试集的词袋模型 
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]: #如果分类错误
            errorCount += 1 #错误数量加1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率: ',float(errorCount)/len(testSet))

from numpy import *
import operator
import sys
import os
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group, labels
def classify0(inx, dataSet, labels, k):
    #计算距离
    dataSetSize=dataSet.shape[0]
        #得到数据集行数
    diffMat=tile(inx,(dataSetSize,1))-dataSet
        #将输入值复制成与数据集 中行数一致 每行一个原数据集中的数据 并计算差值
    sqDiffMat=diffMat**2
    sqDistence=sqDiffMat.sum(axis=1)
        #axis=0表示纵轴相加 axis=1表示横轴相加
    distence=sqDistence**0.5
    sortedDistIndicies=distence.argsort()
        #将distence中的元素从小到大排列，提取其对应的index(索引)，然后输出到sortedDistIndicies
    classCount={}
        #创建一个字典
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]] #找出距离最小的k个数据对应的label
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1#统计对应label的个数
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
        #排序 第一项可迭代类型，key为关键字，reverse true降序 flase升序 这里是统计距离最小的k个数据中label出现的频率
    return sortedClassCount[0][0] #返回最小的第一个域的值 即返回label的值
def file2matrix(filename): #从文本数据中解析数据
    fr=open(filename)
    arrayOLines=fr.readlines() #读取所有行并返回列表
    numberOfLines=len(arrayOLines)#读取行数
    returnMat=zeros((numberOfLines,3))#创建numberOflines维数组，每一个维度3个数值
    ClassLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()   # strip()用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列 截掉回车
        listFromLine=line.split('\t') #指定分割符对字符串切片
        returnMat[index,:] = listFromLine[0:3] #每一行从第0个开始读取3个进入
        ClassLabelVector.append(int(listFromLine[-1]))  # list[-1] 读取列表最后一个值 加到classlabelVector后
        index += 1
    return returnMat,ClassLabelVector
    """
    可视化代码
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy
    fig=plt.figure()
    ax=fig.add_subplot(111)  增加子图  ax = fig.add_subplot(349)参数349的意思是：将画布分割成3行4列，图像画在从左到右从上到下的第9块
    ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
    #ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15*numpy.array(datingLabels),15*numpy.array(datingLabels))
    #后面两个参数调节颜色和大小
    plt.show()
    """
def autoNorm(dataSet):
    #将任意取值范围的特征值转化为0-1区间内的值 归一化数值
    minVals=dataSet.min(0)  #0返回该矩阵中每一列的最小值 1返回每一行中的最小值
    maxVals=dataSet.max(0)  #0返回该矩阵中每一列的最大值 1返回每一行中的最大值
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet)) #构建标准化矩阵
    m=dataSet.shape[0] #m为行数
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))      #特征值相除 将任意取值范围的特征值转化为0-1区间内的值
    return normDataSet,ranges,minVals
def datingClassTest():
    #分类器的测试代码
    hoRatio=0.1 #测试集的比例
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')  #解析文本文件中的数据
    normMat, ranges, minVals=autoNorm(datingDataMat)    #数据归一化
    m=normMat.shape[0]  #数据集的总数
    numTestVecs=int(m*hoRatio)  #测试集的数量
    errorcount=0.0  #分类错误计数器
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        #测试集中的每一个作为输入 数据集中除了测试集之外的数据作为比对的测试集 同理 label k为3
        print("the classifier came back with %d,the real answer is: %d"%(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]): errorcount+=1.0
    print("the total error rate is: %f"%(errorcount/float(numTestVecs)))
    print(int(numTestVecs))
def classifyPerson():
    resultList=['not at all','in small doses', 'in large doses']
    percenTats=float(input("percentage of time spent playing video games?")) #python3 将raw_input 和input整合
    ffMiles=float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr=array([ffMiles,percenTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)#要对输入也进行归一化处理
    print("You will probably like this person:",resultList[classifierResult-1])# 将预测回来的值在resullist中显示出来
def img2vector(filename):
    #将图像格式化成为1x1024
    returnVect=zeros((1,1024))#创建一个空的1x1024向量存储
    fr=open(filename)
    for i in range(32):         #文件中是32x32格式的
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
def handwritingClassTest():
    hwLabels=[]#创建一个空的list
    trainingFileList=os.listdir('trainingDigits')#获取训练数据集文件夹下的文件名称 目录名称
    m=len(trainingFileList) #获取有文件数量
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0] # 用'.'分割 获取前面的 即文件名
        classNumStr=int(fileNameStr.split('_')[0])# 用'_'分割，获取前面的 相当于获取数据的label
        hwLabels.append(classNumStr) # 附加到label后面
        trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr)#通过函数转换 将图片数据转化
    testFileList=os.listdir('testDigits')#获取测试数据集下的文件名称
    errorcount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileNameStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s'%fileNameStr)
        classfierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print("the classifier came back with %d, the real answer is %d"%(classfierResult,classNumStr))
        if (classfierResult!=classNumStr):
            errorcount+=1
    print("\nthe total nbumber of errors is: %d"% errorcount)
    print("\nthe total error rate is: %d"%(errorcount/float(mTest)))










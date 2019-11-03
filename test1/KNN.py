import numpy as np
from math import sqrt
import operator as opt

def normData(dataSet):
    maxVals = dataSet.max(axis=0)
    minVals = dataSet.min(axis=0)
    ranges = maxVals - minVals
    retData = (dataSet - minVals) / ranges
    return retData, ranges, minVals


def kNN(dataSet, labels, testData, k):
    sum_array = np.zeros((testData.shape[0], dataSet.shape[0]))
    for i in range(testData.shape[0]):
        for j in range(dataSet.shape[0]):
            distSquareMat = (dataSet[j] - testData[i]) ** 2 # 计算差值的平方
            distSquareSums = distSquareMat.sum()     # 求差值平方和
            distances = distSquareSums ** 0.5 # 开根号，得出每个样本到测试点的距离
            sum_array[i][j] = distances
    # print(sum_array)
    # 针对每个测试数据，选择k个最邻近，统计标签
    label_array = []
    for i in range(testData.shape[0]):
        one_test = sum_array[i].argsort()[:k]
        labelCount = {} # 存储每个label的出现次数
        for i in one_test:
            label = labels[i]
            labelCount[label] = labelCount.get(label, 0) + 1 # 次数加一
        print(labelCount)
        sortedCount = sorted(labelCount.items(), key=opt.itemgetter(1), reverse=True) # 对label出现的次数从大到小进行排序
        label_array.append(sortedCount[0][0])
    return label_array # 返回出现次数最多的label



if __name__ == "__main__":

    new_data = []
    with open("./shuffle_all_data.txt", "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        new_data.append(eval(data[i]))
    trainXYL = new_data[0:2000]
    testXYL = new_data[-501:-1]
    trainlabel = []
    dataSet = []
    for i in range(len(trainXYL)):
        trainlabel.append(trainXYL[i][1])
        dataSet.append([trainXYL[i][0][0], trainXYL[i][0][1]])
    dataSet = np.asarray(dataSet)
    normDataSet, ranges, minVals = normData(dataSet)
    labels = trainlabel
    testData = []
    testlabels = []
    for i in range(len(testXYL)):
        testData.append([testXYL[i][0][0], testXYL[i][0][1]])
        testlabels.append(testXYL[i][1])
    testData = np.asarray(testData)
    testlabels = np.asarray(testlabels)
    normTestData = (testData - minVals) / ranges
    result = kNN(normDataSet, labels, normTestData, 5)
    print(result)
    # 计算准确率
    num = (np.asarray([result==testlabels])+0).sum()
    print(num)
    accuracy = num/len(result)
    print(accuracy)
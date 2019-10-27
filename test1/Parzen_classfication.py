from mpl_toolkits.mplot3d import Axes3D, axes3d
import math
import random
import matplotlib.pyplot as plt
import numpy as np

'''
    本篇为使用parzon窗方法进行训练和分类。
    1、我理解的训练过程就是使用训练数据来拟合训练数据构建的parzon窗，
    不断更新h和训练数量，以此得到对训练数据拟合最好的图像表示，
    然后使用此时的h和训练数量n,来对测试数据做概率密度预测并分类。
    2、对于一个测试数据，我们指导基本类有几种，将训练数据按类别划分，对每类计算windowsize, 然后计算概率密度。
    3、取该测试数据在所有类别中最大的概率密度，作为它的概率密度。
    4、若先验概率相同（考虑为训练数据的不同类别占总训练数据的概率），则将该数据归类于概率密度最大对应的类别。
    5、若先验概率不同， 则计算后验概率，取最大后验概率分类， 即 MAP 分类方法
'''

# 二维高斯窗
def multi_gausswindow(x):
    # 设置高斯窗为均值为0， 方差为1的二维高斯函数
    u = np.asarray([0, 0])
    sigma = np.asarray([[1, 0], [0, 1]])

    det1 = np.linalg.det(sigma)
    inv1 = np.linalg.inv(sigma)
    c = (x - u).reshape(1, -1)
    dc = c.reshape(-1, 1)
    first = np.dot(c, inv1)
    second = np.dot(first, dc)
    d = x.shape[0]
    p = 1.0 / ((2 * np.pi) ** (d / 2.) * np.sqrt(det1)) * np.exp(-second / 2.)
    return p

def PN(trainXYL, testXYL, h):

    category_num = len(trainXYL)
    test_label = []
    test_x1 = []
    test_x2 = []
    for m in range(len(testXYL)):
        test_label.append(testXYL[m][-1])
        test_x1.append(testXYL[m][0])
        test_x2.append(testXYL[m][1])
    test_label = np.asarray(test_label)
    test_x1 = np.asarray(test_x1)
    test_x2 = np.asarray(test_x2)
    # print("test_label: ", len(test_label))

    predict_label = []
    predict_probability = []
    # 遍历每个测试数据
    for t in range(len(testXYL)):
        # 对每个测试数据推断属于每个类的概率密度
        max_probabi = 0
        classfication = 0
        for i in range(category_num):
            wi = np.asarray(trainXYL[i])
            # windowsize 即 hn
            windowsize = h * 1.0 / np.sqrt(wi.shape[0])
            probability = 0
            for j in range(wi.shape[0]):
                core = (np.asarray([test_x1[t], test_x2[t]])-np.asarray([wi[j][0], wi[j][1]]))/windowsize
                one = multi_gausswindow(core) / windowsize
                probability += one
            probability = probability / wi.shape[0]
            # print('test older: ', t, 'category: ', i, ',  probability: ', probability[0][0])
            if probability[0][0] > max_probabi:
                max_probabi = probability[0][0]
                classfication = i

        predict_label.append(classfication)
        predict_probability.append(max_probabi)

    predict_probability = np.asarray(predict_probability, dtype=np.float)
    predict_label = np.asarray(predict_label)
    # print('predicr_shape: ', predict_probability.shape)
    predict_accuracy = calculate_acc(predict_label, test_label)
    print('predict accuracy is: ', predict_accuracy)
    return [test_x1, test_x2, predict_probability, predict_accuracy]




# 3d表面形状绘制
def surface_3d(x1,x2,z, img_path, h, num_train, p_accuracy):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x1, x2, z, cmap=plt.get_cmap('YlOrRd'))
    title = 'h: '+str(h)+', num_train: '+str(num_train)+', accuracy: '+str(p_accuracy)
    ax.set_title(label=title)
    plt.savefig(img_path)
    # plt.show()
    plt.close()

def calculate_acc(predict, labels):
    num = np.sum((predict==labels)+0)
    return num / np.float(labels.shape[0])



if __name__ == '__main__':
    # hh为计算窗口大小的参数
    hh = [0.1, 0.5, 1, 2, 5, 10]
    # num_train为每类取的训练样本数量, num_test为测试使用的总样本数量
    num_train = [3, 10, 100, 500, 1000]
    num_test = 300

    # 读取生成的训练数据
    w1 = []
    w2 = []
    w3 = []
    trainXYL = []
    with open("./w1_data.txt", "r") as f1:
        data1 = f1.readlines()
    for i in range(0, len(data1)):
        w1.append(eval(data1[i]))

    with open("./w2_data.txt", "r") as f2:
        data2 = f2.readlines()
    for i in range(0, len(data2)):
        w2.append(eval(data2[i]))

    with open("./w3_data.txt", "r") as f3:
        data3 = f3.readlines()
    for i in range(0, len(data3)):
        w3.append(eval(data3[i]))


    testXYL = []
    with open('./shuffle_test_data.txt', "r") as f4:
        data4 = f4.readlines()
    for i in range(0, len(data4)):
        testXYL.append(eval(data4[i]))


    for h in hh:
        for nnn in num_train:
            trainXYL = [w1[:nnn], w2[:nnn], w3[:nnn]]
            result = PN(trainXYL, testXYL[:num_test], h)
            print('h: %f, nnn: %d is over!' % (h, nnn))
            img_path = './test_show/img_'+str(h)+'_'+str(nnn)+'.jpg'
            surface_3d(result[0], result[1], result[2], img_path, h, nnn, result[3])

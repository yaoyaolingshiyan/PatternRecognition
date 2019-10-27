'''
    这里尝试对训练数据并不区分类别，直接求概率密度
    （不对训练数据分类求概率密度是不对的，这里仅仅看一下效果，个人好奇的原因）
'''

from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import numpy as np


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

def PN(trainXYL, h):
    # windowsize 即 hn
    windowsize = h * 1.0 / np.sqrt(len(trainXYL))
    predict_probability = []
    x1 = []
    x2 = []
    for n in range(len(trainXYL)):
        x1.append(trainXYL[n][0])
        x2.append(trainXYL[n][1])
    # 遍历每个测试数据
    for t in range(len(trainXYL)):
        probability = 0
        for i in range(len(trainXYL)):
            core = (np.asarray([x1[t], x2[t]]) - np.asarray([x1[i], x2[i]])) / windowsize
            one = multi_gausswindow(core) / windowsize
            probability += one
        predict_probability.append((probability / len(trainXYL))[0][0])
    predict_probability = np.asarray(predict_probability, dtype=np.float)
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    return [x1, x2, predict_probability]




# 3d表面形状绘制
def surface_3d(x1,x2,z, img_path, h, num_train):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x1, x2, z, cmap=plt.get_cmap('YlOrRd'))
    title = 'h: '+str(h)+', num_train: '+str(num_train)
    ax.set_title(label=title)
    plt.savefig(img_path)
    # plt.show()
    plt.close()



if __name__ == '__main__':
    # hh为计算窗口大小的参数
    hh = [0.1, 0.5, 1, 2, 5, 10]
    # num_train为取的训练样本数量
    num_train = [3, 10, 100, 500, 1000]

    # 读取生成的训练数据
    trainXYL = []
    with open("./shuffle_all_data.txt", "r") as f1:
        data1 = f1.readlines()
    for i in range(0, len(data1)):
        trainXYL.append(eval(data1[i]))


    for h in hh:
        for nnn in num_train:
            result = PN(trainXYL[:nnn], h)
            # print(result)
            print('h: %f, nnn: %d is over!' %(h, nnn))
            img_path = './without_classfication_show/img_'+str(h)+'_'+str(nnn)+'.jpg'
            surface_3d(result[0], result[1], result[2], img_path, h, nnn)

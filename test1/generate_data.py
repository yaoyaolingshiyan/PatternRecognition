from mpl_toolkits.mplot3d import Axes3D, axes3d
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# 一维高斯分布求Y
def one_dim_z(x, u, sigma):
    z = 1.0/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-u)**2/(2*sigma*sigma))
    return

# 多维高斯分布求Z, 维度>=2, 使用协方差矩阵
def large_dim_z(x, u, sigma):
    # det1 是协方差矩阵的行列式
    det1 = np.linalg.det(sigma)
    inv1 = np.linalg.inv(sigma)
    c = (x - u).reshape(1, -1)
    dc = c.reshape(-1, 1)
    first = np.dot(c, inv1)
    second = np.dot(first, dc)
    d = len(x)
    p = 1.0/((2*np.pi)**(d/2.)*np.sqrt(det1))*np.exp(-second/2.)
    return p

def get_data(u, sigma, type, num_sample):
    s = np.random.multivariate_normal(u, sigma, num_sample)
    z = []
    for x in s:
        pp = large_dim_z(x, u, sigma)[0][0]
        z.append([x[0], x[1], pp, type])
    return z

# 生成训练数据
def generate_data():
    # 每类数据生成的样本数量
    num_sample = 1000
    # 二维高斯分布的均值和协方差矩阵如下：
    u1 = np.asarray([0, 0])
    sigma1 = np.asarray([[1, 0], [0, 1]])

    u2 = np.asarray([2, 0])
    sigma2 = np.asarray([[1, 0], [0, 1]])

    u3 = np.asarray([4, 0])
    sigma3 = np.asarray([[1, 0], [0, 1]])

    w1 = get_data(u1, sigma1, 0, num_sample)
    w2 = get_data(u2, sigma2, 1, num_sample)
    w3 = get_data(u3, sigma3, 2, num_sample)


    with open("./w1_data.txt", "w") as f1:
        for i in range(len(w1)):
            f1.write(str(w1[i])+'\n')

    with open("./w2_data.txt", "w") as f2:
        for i in range(len(w2)):
            f2.write(str(w2[i])+'\n')

    with open("./w3_data.txt", "w") as f3:
        for i in range(len(w3)):
            f3.write(str(w3[i])+'\n')

    all_data = w1
    all_data.extend(w2)
    all_data.extend(w3)
    random.shuffle(all_data)
    with open("./shuffle_all_data.txt", "w") as f4:
        for i in range(len(all_data)):
            f4.write(str(all_data[i])+'\n')


# 生成的测试数据必须属于训练数据的三类，所以均值和协方差矩阵相同
def generate_test_data():

    # 每类数据生成的测试数据
    num_samples = 100
    # 二维高斯分布的均值和协方差矩阵如下：
    u1 = np.asarray([0, 0])
    sigma1 = np.asarray([[1, 0], [0, 1]])

    u2 = np.asarray([2, 0])
    sigma2 = np.asarray([[1, 0], [0, 1]])

    u3 = np.asarray([4, 0])
    sigma3 = np.asarray([[1, 0], [0, 1]])

    w1 = get_data(u1, sigma1, 0, num_samples)
    w2 = get_data(u2, sigma2, 1, num_samples)
    w3 = get_data(u3, sigma3, 2, num_samples)

    all_data = w1
    all_data.extend(w2)
    all_data.extend(w3)
    random.shuffle(all_data)
    with open("./shuffle_test_data.txt", "w") as f4:
        for i in range(len(all_data)):
            f4.write(str(all_data[i])+'\n')

# 读取数据并打印
def read_data(data_path):
    shuffle_data = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        shuffle_data.append(eval(data[i]))
    print(shuffle_data)
    print('samples number are: ', len(shuffle_data))

# 绘制数据分布图象
def draw_data(data_path, save_name):
    all_data = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        all_data.append(eval(data[i]))

    x1 = []
    x2 = []
    z = []
    for i in range(len(all_data)):
        x1.append(all_data[i][0])
        x2.append(all_data[i][1])
        z.append(all_data[i][2])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_trisurf(x1, x2, z, cmap=plt.get_cmap('YlOrRd'))
    img_path = './data_show/'+save_name
    plt.savefig(img_path)
    plt.show()


if __name__ == '__main__':
    print('---project start---')
    # 生成存储结果的文件夹
    path1 = './classfication_show'
    path2 = './data_show'
    path3 = './test_show'
    path4 = './without_classfication_show'
    all_path = [path1, path2, path3, path4]
    for pa in all_path:
        if not os.path.exists(pa):
            os.makedirs(pa)
        else:
            print(pa, ' have existed!')

    # 生成数据
    # generate_data()
    # generate_test_data()
    #
    # train_data_path = './shuffle_all_data.txt'
    # test_data_path = './shuffle_test_data.txt'
    # # 读取数据
    # # read_data(train_data_path)
    # # 绘制数据分布图
    # draw_data(train_data_path, save_name='train_data.jpg')
    # draw_data(test_data_path, save_name='test_data.jpg')
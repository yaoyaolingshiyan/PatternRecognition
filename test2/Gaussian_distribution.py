import numpy as np
import random
import matplotlib.pyplot as plt

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

def get_data(u, sigma, type):
    num_sample = 1000
    s = np.random.multivariate_normal(u, sigma, num_sample)
    z = []
    for x in s:
        result = large_dim_z(x, u, sigma)
        # [x1, x2, x3, z]
        z.append([x[0], x[1], x[2], result[0][0], type])
    return z

def generate_data():
    # 四组三维高斯分布的均值和协方差矩阵如下：
    u1 = np.asarray([0, 0, 0])
    sigma1 = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    u2 = np.asarray([0, 1, 0])
    sigma2 = np.asarray([[1, 0, 1], [0, 2, 2], [1, 2, 5]])

    u3 = np.asarray([-1, 0, 1])
    sigma3 = np.asarray([[2, 0, 0], [0, 6, 0], [0, 0, 1]])

    u4 = np.asarray([0, 0.5, 1])
    sigma4 = np.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 3]])

    result = get_data(u1, sigma1, 0)
    result.extend(get_data(u2, sigma2, 1))
    result.extend(get_data(u3, sigma3, 2))
    result.extend(get_data(u4, sigma4, 3))

    with open("./data.txt", "w") as f1:
        for i in range(len(result)):
            f1.write(str(result[i])+'\n')

    random.shuffle(result)

    with open("./shuffle_data.txt", "w") as f2:
        for i in range(len(result)):
            f2.write(str(result[i])+'\n')

def read_data():
    data_path = './shuffle_data.txt'
    shuffle_data = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        shuffle_data.append(eval(data[i]))
    print(shuffle_data)
    print(len(shuffle_data))
if __name__ == '__main__':
    print('hello')
    # 生成数据
    generate_data()
    # 读取数据
    read_data()
from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import numpy as np

# 多维高斯分布求Z, 维度>=2, 使用协方差矩阵
def large_dim_z(x, u, sigma):
    # print("calcuing...")
    # det1 是协方差矩阵的行列式
    det1 = np.linalg.det(sigma)
    inv1 = np.linalg.inv(sigma)
    c = (x - u).reshape(1, -1)
    dc = c.reshape(-1, 1)
    first = np.dot(c, inv1)
    second = np.dot(first, dc)
    d = len(x)
    p = 1.0/((2*np.pi)**(d/2.)*np.sqrt(det1))*np.exp(-second/2.)
    # print('one is over!')
    return p


# 3d表面形状绘制
def surface_3d():

    data_path = './2d_data.txt'
    shuffle_data = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        shuffle_data.append(eval(data[i]))
    x1 = []
    x2 = []
    for i in range(len(shuffle_data)):
        x1.append(shuffle_data[i][0])
        x2.append(shuffle_data[i][1])

    print(len(x1))
    print(len(x2))
    x = np.asarray(x1).reshape((20, 20))
    y = np.asarray(x2).reshape((20, 20))
    z = np.zeros_like(x)
    print('x:', x.shape)
    print('y:', y.shape)
    print('z:', z.shape)
    # 生成数据的均值和协方差矩阵
    u1 = np.asarray([0, 0])
    sigma1 = np.asarray([[1, 0], [0, 1]])
    print('hello')
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            print('i : %d,  j: %d' %(i, j))
            z[i][j] = large_dim_z(np.asarray([x[i][j], y[i][j]]), u1, sigma1)
    print('calcu over!')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=plt.get_cmap('Blues'))
    plt.show()

# 3d直线绘制
def line_3d():
    data_path = './2d_data.txt'
    shuffle_data = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        shuffle_data.append(eval(data[i]))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x1 = []
    x2 = []
    z1 = []
    for i in range(len(shuffle_data)):
        x1.append(shuffle_data[i][0])
        x2.append(shuffle_data[i][1])
        z1.append(shuffle_data[i][2])

    # theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    # z = np.linspace(-2, 2, 100)
    # r = z ** 2 + 1
    # x = r * np.sin(theta)
    # y = r * np.cos(theta)
    # print('z: ', z.shape)

    ax.plot(x1, x2, z1, label='parametric curve')
    ax.legend()
    plt.show()

# 3d直方图绘制
def histogram_3d():
    data_path = './2d_data.txt'
    shuffle_data = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        shuffle_data.append(eval(data[i]))
    x1 = []
    x2 = []
    for i in range(len(shuffle_data)):
        x1.append(shuffle_data[i][0])
        x2.append(shuffle_data[i][1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(x1, x2, bins=4)
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
    print('xpos: ', xpos)
    print('ypos: ', ypos)
    xpos = xpos.flatten('F')
    ypos = ypos.flatten('F')
    zpos = np.zeros_like(xpos)

    dx = 0.5*np.ones_like(zpos)
    # print('dx: ', dx)
    dy = dx.copy()
    dz = hist.flatten()

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    plt.show()



# 3d网格图绘制
def grid_3d():

    data_path = './2d_data.txt'
    shuffle_data = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        shuffle_data.append(eval(data[i]))
    x1 = []
    x2 = []
    for i in range(len(shuffle_data)):
        x1.append(shuffle_data[i][0])
        x2.append(shuffle_data[i][1])

    print(len(x1))
    print(len(x2))
    x = np.asarray(x1).reshape((20, 20))
    y = np.asarray(x2).reshape((20, 20))
    z = np.zeros_like(x)
    print('x:', x.shape)
    print('y:', y.shape)
    print('z:', z.shape)

    # 生成数据的均值和协方差矩阵
    u1 = np.asarray([0, 0])
    sigma1 = np.asarray([[1, 0], [0, 1]])
    print('hello')
    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            print('i : %d,  j: %d' %(i, j))
            z[i][j] = large_dim_z(np.asarray([x[i][j], y[i][j]]), u1, sigma1)
    print('calcu over!')


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot a basic wireframe
    ax.plot_wireframe(x, y, z, rstride=1, cstride=1)
    plt.show()


if __name__ == '__main__':
    # surface_3d()
    # line_3d()
    # histogram_3d()
    grid_3d()
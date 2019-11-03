import numpy as np
from matplotlib import pyplot as plt
import random


# 多维高斯分布求Z, 维度>=2, 使用协方差矩阵
# 计算理论条件概率
def large_dim_z(x, u, sigma):
    x = np.asarray(x)
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


def softmax(m):
    exps = np.exp(m)
    return exps/np.sum(exps)


def stable_softmax(m):
    exps = np.exp(m-np.max(m))
    return exps/np.sum(exps)


def mean_square_sum(predict, label):

    return np.sum((label-predict)**2) / 2.0

def read_data(data_path):

    shuffle_data = []
    label_list = []
    x_list = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        shuffle_data.append(eval(data[i]))
    for i in range(len(shuffle_data)):
        x_list.append([shuffle_data[i][0], shuffle_data[i][1], shuffle_data[i][2]])
        label_list.append(shuffle_data[i][3])

    one_hot_label = []
    for i in range(len(label_list)):
        if int(label_list[i]) == 0:
            one_hot_label.append([1, 0, 0, 0])
        elif int(label_list[i]) == 1:
            one_hot_label.append([0, 1, 0, 0])
        elif int(label_list[i]) == 2:
            one_hot_label.append([0, 0, 1, 0])
        elif int(label_list[i]) == 3:
            one_hot_label.append([0, 0, 0, 1])
        else:
            raise KeyError

    return x_list, one_hot_label


def network(x_list, label_list):

    # 训练停止阈值
    threshold = 0.0001
    # 训练损失变化
    plt_list = []
    # 测试准确率变化
    test_acc_list = []
    epochs = 80
    batch = 32
    learn_rate_list = [0.01, 0.001, 0.0001]
    # 初始化参数
    w1 = np.random.rand(3, 3)
    bias1 = np.random.rand(3)
    w2 = np.random.rand(3, 4)
    bias2 = np.random.rand(4)
    # print('w1: ', w1.shape)
    # print('w2: ', w2.shape)

    for epoch in range(epochs):
        if epoch < 60:
            learn_rate = learn_rate_list[0]
        elif epoch >= 60 and epoch < 200:
            learn_rate = learn_rate_list[1]
        else:
            learn_rate = learn_rate_list[2]

        randnum = random.randint(0, len(x_list))
        random.seed(randnum)
        random.shuffle(x_list)
        random.seed(randnum)
        random.shuffle(label_list)

        loss = 0
        # 权值变化矩阵
        dw2 = np.zeros((3, 4))
        dw1 = np.zeros((3, 3))

        predict_list = []
        for step in range(len(x_list)):
            x = np.asarray(x_list[step])
            label = np.asarray(label_list[step])
            # print('x:', x)
            # forward
            layer1 = np.dot(x, w1)+bias1
            # relu激活函数
            layer1_tanh = np.ones(3)
            for it in range(3):
                if layer1[it] >= 0:
                    layer1_tanh[it] = layer1[it]
                else:
                    layer1_tanh[it] = 0
            layer2 = np.dot(layer1_tanh, w2)+bias2
            result = stable_softmax(layer2)
            predict = result.tolist()
            nn = predict.index(max(predict))
            predict_list.append(nn)

            # 每个epoch平均损失
            loss += mean_square_sum(result, label)

            # 以下计算反向传播
            # softmax层导数
            dsoft = np.zeros(4)
            for k in range(4):
                dsoft[k] = np.asarray(result[k] * (1 - result[k]))

            # 输出层敏感度
            ses = np.zeros(4)
            for k2 in range(4):
                ses[k2] = (label[k2] - result[k2]) * dsoft[k2]

            # 隐含层到输出层权值更新
            for k3 in range(4):  # k3表示种类数, 每行
                for k4 in range(3):
                    dw2[k4, k3] += learn_rate * ses[k3] * layer1_tanh[k4]

            # 隐含层敏感度
            ses2 = np.zeros(3)
            for k5 in range(3):
                # 第一层使用tanh激活函数
                summ = 0
                for i in range(4):
                    summ += w2[k5, i]*ses[i]
                if layer1[k5] >= 0:
                    ses2[k5] = summ
                else:
                    ses2[k5] = 0

            # 输出层到隐含层权值更新
            for k6 in range(3):
                for k7 in range(3):
                    dw1[k7, k6] += learn_rate * ses2[k6] * x[k7]
            if step % batch == 0:
                # 更新参数
                w1 = w1 + dw1
                w2 = w2 + dw2
                bias1 = bias1 + learn_rate*ses2
                bias2 = bias2 + learn_rate*ses
                # 反向传播
                dw2 = np.zeros((3, 4))
                dw1 = np.zeros((3, 3))



        # train_accuracy
        accuracy = calcu_accuracy(predict_list, label_list)
        # print('dw1: ', dw1)

        print('epochs: %d , loss= %f, accuracy: %f' % (epoch+1, loss / len(x_list), accuracy))
        # 测试
        test_x_array, test_label_array = read_data(test_data_path)
        predict_li = network_test(test_x_array, w1, w2, bias1, bias2)
        test_accuracy = calcu_accuracy(predict_li, test_label_array)
        print('epoch: %d , test accuracy: %f' % (epoch, test_accuracy))
        test_acc_list.append([epoch, test_accuracy])
        plt_list.append([epoch, loss / len(x_list), accuracy])
        if loss / len(x_list) < threshold:
            return [w1, w2, bias1, bias2, plt_list, test_acc_list]

    return [w1, w2, bias1, bias2, plt_list, test_acc_list]

def calcu_accuracy(predict_list, label_list):
    labels = []
    for a in range(len(label_list)):
        # print('mmm: ', mmm)
        if label_list[a] == [0, 0, 0, 1]:
            labels.append(3)
        elif label_list[a] == [0, 0, 1, 0]:
            labels.append(2)
        elif label_list[a] == [0, 1, 0, 0]:
            labels.append(1)
        elif label_list[a] == [1, 0, 0, 0]:
            labels.append(0)
        else:
            raise KeyError("there don't have the label!")


    predict = np.asarray(predict_list)
    labels = np.asarray(labels)

    # print('predict: ', predict)
    # print('labels: ', labels.shape[0])
    num = np.sum((predict == labels) + 0)
    # print(num)
    return num / np.float(labels.shape[0])


# x_list, w1, w2, bias1, bias2
# 训练过程中使用的测试方式，需要输入训练的权重
def network_test(x_list, w1, w2, bias1, bias2):
    # 初始化参数
    # w1_path = './train_param/train_w1.txt'
    # w2_path = './train_param/train_w2.txt'
    # bias1_path = './train_param/train_bias1.txt'
    # bias2_path = './train_param/train_bias2.txt'
    #
    # w1 = np.loadtxt(w1_path)
    # w2 = np.loadtxt(w2_path)
    # bias1 = np.loadtxt(bias1_path)
    # bias2 = np.loadtxt(bias2_path)


    predict_list = []
    for step in range(len(x_list)):
        x = np.asarray(x_list[step])
        # forward
        layer1 = np.dot(x, w1) + bias1
        # relu激活函数
        layer1_tanh = np.ones(3)
        for it in range(3):
            if layer1[it] >= 0:
                layer1_tanh[it] = layer1[it]
            else:
                layer1_tanh[it] = 0
        layer2 = np.dot(layer1_tanh, w2) + bias2
        result = stable_softmax(layer2)
        predict = result.tolist()
        nn = predict.index(max(predict))
        predict_list.append(nn)
    return predict_list


# 单独测试时使用的测试方式，权重从保存文件中读取
def network_test_without_train(x_list):
    # 初始化参数
    w1_path = './train_param/train_w1.txt'
    w2_path = './train_param/train_w2.txt'
    bias1_path = './train_param/train_bias1.txt'
    bias2_path = './train_param/train_bias2.txt'

    w1 = np.loadtxt(w1_path)
    w2 = np.loadtxt(w2_path)
    bias1 = np.loadtxt(bias1_path)
    bias2 = np.loadtxt(bias2_path)


    predict_list = []
    probabi = []
    for step in range(len(x_list)):
        x = np.asarray(x_list[step])
        # forward
        layer1 = np.dot(x, w1) + bias1
        # relu激活函数
        layer1_tanh = np.ones(3)
        for it in range(3):
            if layer1[it] >= 0:
                layer1_tanh[it] = layer1[it]
            else:
                layer1_tanh[it] = 0
        layer2 = np.dot(layer1_tanh, w2) + bias2
        result = stable_softmax(layer2)
        predict = result.tolist()
        probabi.append(predict)
        nn = predict.index(max(predict))
        predict_list.append(nn)
    return probabi, predict_list

# 画出训练损失和训练准确率变化折线图
def draw_line():
    data_path = './train_param/train_plt_data.txt'
    plt_data = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        plt_data.append(eval(data[i]))

    x = []
    y = []
    y2 = []
    # print(plt_data)
    for m in range(len(plt_data)):
        x.append(plt_data[m][0])
        y.append(plt_data[m][1])
        y2.append(plt_data[m][2])
    # print(x)
    # print(y)
    plt.title('train')
    plt.xlabel('epoch')
    plt.ylabel('result')
    plt.plot(x, y, color='g', label='loss')
    plt.plot(x, y2, color='r', label='accuracy')
    plt.legend(loc='upper left')
    plt.show()
    plt.close()


# 画出测试集准确率变化折线图
def draw_acc():
    data_path = "./train_param/test_plt_data.txt"
    plt_data = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        plt_data.append(eval(data[i]))

    x = []
    y = []
    # print(plt_data)
    for m in range(len(plt_data)):
        x.append(plt_data[m][0])
        y.append(plt_data[m][1])
    # print(x)
    # print(y)
    plt.title('test accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(x, y)
    plt.show()
    plt.close()

if __name__ == '__main__':
    # train_data_path = './train_data/train_shuffle_data.txt'
    # test_data_path = './test_data/test_shuffle_data.txt'
    # # 读取数据
    # x_list, label_list = read_data(train_data_path)
    # test_x_list, test_label_list = read_data(test_data_path)


    # return_result = network(x_list, label_list)
    # pl_list = return_result[4]
    # test_accu_list = return_result[5]
    #
    # # 保存参数
    # w1_path = './train_param/train_w1.txt'
    # w2_path = './train_param/train_w2.txt'
    # bias1_path = './train_param/train_bias1.txt'
    # bias2_path = './train_param/train_bias2.txt'
    # np.savetxt(w1_path, return_result[0])
    # np.savetxt(w2_path, return_result[1])
    # np.savetxt(bias1_path, return_result[2])
    # np.savetxt(bias2_path, return_result[3])
    # with open("./train_param/train_plt_data.txt", "w") as f1:
    #     for i1 in range(len(pl_list)):
    #         f1.write(str(pl_list[i1])+'\n')
    #
    # with open("./train_param/test_plt_data.txt", "w") as f2:
    #     for i2 in range(len(test_accu_list)):
    #         f2.write(str(test_accu_list[i2])+'\n')

    # 使用保存的训练好的权重单独测试

    # 对给定数据做预测
    test_data = [[0, 0, 0], [-1, 0, 1], [0.5, -0.5, 0], [-1, 0, 0], [0, 0, -1]]
    probability, predict_li = network_test_without_train(test_data)
    # print("predict_li: ", predict_li)

    # accuracy = calcu_accuracy(predict_li, test_label_list)
    # print('test accuracy: ', accuracy)

    # 计算理论条件概率
    # 四组三维高斯分布的均值和协方差矩阵如下：
    u1 = np.asarray([0, 0, 0])
    sigma1 = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    u2 = np.asarray([0, 1, 0])
    sigma2 = np.asarray([[1, 0, 1], [0, 2, 2], [1, 2, 5]])

    u3 = np.asarray([-1, 0, 1])
    sigma3 = np.asarray([[2, 0, 0], [0, 6, 0], [0, 0, 1]])

    u4 = np.asarray([0, 0.5, 1])
    sigma4 = np.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 3]])

    condi = []
    for i in range(len(test_data)):
        condi.append([large_dim_z(test_data[i], u1, sigma1),
                      large_dim_z(test_data[i], u2, sigma2),
                      large_dim_z(test_data[i], u3, sigma3),
                      large_dim_z(test_data[i], u4, sigma4)])
    # 理论计算得到的标签
    calcu_label = []
    for i in range(5):

        condipro = np.asarray(condi[i]).reshape(4)
        print('condi: ', condipro)
        print('houyangailv: ', condipro/np.sum(condipro))
        print('probability: ',  probability[i])
        calcu_label.append(np.where(np.max(condipro/np.sum(condipro))))
        print('---------', i, ' is over!-------')
    print('神经网络推断得到标签：', predict_li)
    print('理论计算得到标签：', calcu_label)





    # 绘制训练loss和准确率
    # draw_line()
    # 绘制训练过程中测试集的准确率变化
    # draw_acc()

import numpy as np
from sklearn import preprocessing


def softmax(m):


    exps = np.exp(m)
    return exps/np.sum(exps)


def stable_softmax(m):
    exps = np.exp(m-np.max(m))
    return exps/np.sum(exps)


def cross_entropy(predict, label):

    return -np.sum(label*(np.log(predict)))


def network(x):

    w1 = np.random.rand(3,3)
    bias1 = np.random.rand(3).reshape(1, -1)
    w2 = np.random.rand(3, 3)
    bias2 = np.random.rand(3).reshape(1, -1)
    w3 = np.random.rand(3, 4)
    bias3 = np.random.rand(4).reshape(1, -1)


    layer1 = np.dot(x, w1)+bias1
    layer2 = np.dot(layer1, w2)+bias2
    layer3 = np.dot(layer2, w3) + bias3
    result = stable_softmax(layer3)
    return result


def read_data():
    data_path = './shuffle_data.txt'
    shuffle_data = []
    label_list = []
    x_list = []
    with open(data_path, "r") as f:
        data = f.readlines()
    for i in range(0, len(data)):
        shuffle_data.append(eval(data[i]))
    for i in range(len(shuffle_data)):
        x_list.append([shuffle_data[i][0], shuffle_data[i][1], shuffle_data[i][2]])
        label_list.append(shuffle_data[i][-1])

    new_x = np.asarray(x_list)
    label = np.asarray(label_list).reshape(-1, 1)
    one_hot = preprocessing.OneHotEncoder(sparse=False, categories='auto')
    new_label = one_hot.fit_transform(label)
    return new_x, new_label

if __name__ == '__main__':

    steps = 5
    x_array, label_array = read_data()
    for step in range(1, steps+1):
        for i in range(len(x_array)):
            x = x_array[i]
            label = label_array[i]
            predict = network(x)
            loss = cross_entropy(predict, label)
            print('step: %d : %d, loss= %f' %(step, i, loss))

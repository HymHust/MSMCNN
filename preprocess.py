from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing  # 0-1编码
from sklearn.model_selection import StratifiedShuffleSplit  # 随机划分，保证每一类比例相同


def prepro(d_path, length=864, number=1000, normal=True, rate=[0.5, 0.25, 0.25], enc=True, enc_step=28):

    # 获得该文件夹下所有.mat文件名
    filenames = os.listdir(d_path)

    def capture(original_path):

        files = {}
        for i in filenames:
            # 文件路径
            file_path = os.path.join(d_path, i)
            file = loadmat(file_path)
            file_keys = file.keys()
            for key in file_keys:
                if 'DE' in key:
                    files[i] = file[key].ravel()
        return files

    def slice_enc(data, slice_rate=rate[1] + rate[2]):

        keys = data.keys()
        Train_Samples = {}
        Test_Samples = {}
        for i in keys:
            slice_data = data[i]
            all_lenght = len(slice_data)
            end_index = int(all_lenght * (1 - slice_rate))
            samp_train = int(number * (1 - slice_rate))  # 700
            Train_sample = []
            Test_Sample = []
            if enc:
                enc_time = length // enc_step
                samp_step = 0  # 用来计数Train采样次数
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - 2 * length))
                    label = 0
                    for h in range(enc_time):
                        samp_step += 1
                        random_start += enc_step
                        sample = slice_data[random_start: random_start + length]
                        Train_sample.append(sample)
                        if samp_step == samp_train:
                            label = 1
                            break
                    if label:
                        break
            else:
                for j in range(samp_train):
                    random_start = np.random.randint(low=0, high=(end_index - length))
                    sample = slice_data[random_start:random_start + length]
                    Train_sample.append(sample)

            # 抓取测试数据
            for h in range(number - samp_train):
                random_start = np.random.randint(low=end_index, high=(all_lenght - length))
                sample = slice_data[random_start:random_start + length]
                Test_Sample.append(sample)
            Train_Samples[i] = Train_sample
            Test_Samples[i] = Test_Sample
        return Train_Samples, Test_Samples

    # 仅抽样完成，打标签
    def add_labels(train_test):
        X = []
        Y = []
        label = 0
        for i in filenames:
            x = train_test[i]
            X += x
            lenx = len(x)
            Y += [label] * lenx
            label += 1
        return X, Y

    # one-hot编码
    def one_hot(Train_Y, Test_Y):
        Train_Y = np.array(Train_Y).reshape([-1, 1])
        Test_Y = np.array(Test_Y).reshape([-1, 1])
        Encoder = preprocessing.OneHotEncoder()
        Encoder.fit(Train_Y)
        Train_Y = Encoder.transform(Train_Y).toarray()
        Test_Y = Encoder.transform(Test_Y).toarray()
        Train_Y = np.asarray(Train_Y, dtype=np.int32)
        Test_Y = np.asarray(Test_Y, dtype=np.int32)
        return Train_Y, Test_Y

    def scalar_stand(Train_X, Test_X):
        # 用训练集标准差标准化训练集以及测试集
        scalar = preprocessing.StandardScaler().fit(Train_X)
        Train_X = scalar.transform(Train_X)
        Test_X = scalar.transform(Test_X)
        return Train_X, Test_X

    def valid_test_slice(Test_X, Test_Y):
        test_size = rate[2] / (rate[1] + rate[2])
        ss = StratifiedShuffleSplit(n_splits=1, test_size=test_size)
        for train_index, test_index in ss.split(Test_X, Test_Y):
            X_valid, X_test = Test_X[train_index], Test_X[test_index]
            Y_valid, Y_test = Test_Y[train_index], Test_Y[test_index]
            return X_valid, Y_valid, X_test, Y_test

    # 从所有.mat文件中读取出数据的字典
    data = capture(original_path=d_path)
    # 将数据切分为训练集、测试集
    train, test = slice_enc(data)
    # 为训练集制作标签，返回X，Y
    Train_X, Train_Y = add_labels(train)
    # 为测试集制作标签，返回X，Y
    Test_X, Test_Y = add_labels(test)
    # 为训练集Y/测试集One-hot标签
    Train_Y, Test_Y = one_hot(Train_Y, Test_Y)
    # 训练数据/测试数据 是否标准化.
    if normal:
        Train_X, Test_X = scalar_stand(Train_X, Test_X)
    else:
        # 需要做一个数据转换，转换成np格式.
        Train_X = np.asarray(Train_X)
        Test_X = np.asarray(Test_X)
    # 将测试集切分为验证集合和测试集.
    Valid_X, Valid_Y, Test_X, Test_Y = valid_test_slice(Test_X, Test_Y)
    return Train_X, Train_Y, Valid_X, Valid_Y, Test_X, Test_Y


if __name__ == "__main__":
    path = r'data\0HP'
    train_X, train_Y, valid_X, valid_Y, test_X, test_Y = prepro(d_path=path,
                                                                length=864,
                                                                number=1000,
                                                                normal=False,
                                                                rate=[0.5, 0.25, 0.25],
                                                                enc=False,
                                                                enc_step=28)
    print(train_Y)
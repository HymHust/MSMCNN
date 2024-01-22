import preprocess
from tensorflow.keras.optimizers import Nadam
from tensorflow.keras.callbacks import TensorBoard,ModelCheckpoint
from keras.models import load_model
from keras.layers import Conv1D, Dense, Dropout, LeakyReLU,BatchNormalization, MaxPooling1D, GlobalMaxPool1D, Flatten,Input,concatenate,Activation
from keras.models import Model
from keras import layers
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model
import numpy as np
#import hunxiao
# 训练参数
batch_size = 128
epochs = 20
num_classes = 10
length = 2560
BatchNorm = True # 是否批量归一化
number = 1000 # 每类样本的数量
normal = True # 是否标准化
rate = [0.7,0.2,0.1] # 测试集验证集划分比例
excel={}
accuracylist=[]
accuracycross=[]
path =r".\data\0HP"


def MSM(filters, kernerl_size, strides, conv_padding, pool_padding, arf, pool_size, dil_rate, inputs):
    x = Conv1D(filters=16, kernel_size=64, strides=8,
               padding='same', kernel_regularizer=l2(1e-4))(inputs)

    x = (LeakyReLU(alpha=arf))(x)
    x1 = MaxPooling1D(pool_size=pool_size, padding=pool_padding)(x)
    x = Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides,
               padding=conv_padding, dilation_rate=dil_rate, kernel_regularizer=l2(1e-3))(x1)
    x = (LeakyReLU(alpha=arf))(x)
    x = MaxPooling1D(pool_size=pool_size, padding=pool_padding)(x)
    model = Model(inputs=inputs, outputs=x)
    return model


def DCM(filters, kernerl_size, strides, conv_padding, pool_padding, arf, pool_size, BatchNormal,
          inputs):
    x = Conv1D(filters=filters, kernel_size=kernerl_size, strides=strides,
               padding=conv_padding, kernel_regularizer=l2(1e-4))(inputs)
    if BatchNormal:
        x = BatchNormalization()(x)
    x = (LeakyReLU(alpha=arf))(x)
    x = MaxPooling1D(pool_size=pool_size, padding=pool_padding)(x)
    return x

for i in range(1,10):
    number = 1000
    x_train, y_train, x_valid, y_valid, x_test, y_test = preprocess.prepro(d_path=path,length=length,
                                                                      number=number,
                                                                      normal=normal,
                                                                      rate=rate,
                                                                      enc=True, enc_step=28)

    # 输入数据的维度
    x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]
    input_shape =x_train.shape[1:]

    print('训练样本维度:', x_train.shape)
    print(x_train.shape[0], '训练样本个数')
    print('验证样本的维度', x_valid.shape)
    print(x_valid.shape[0], '验证样本个数')
    print('测试样本的维度', x_test.shape)
    print(x_test.shape[0], '测试样本个数')
    inputs = Input(shape=input_shape)
    modelA = MSM(filters=16, kernerl_size=3, strides=1, conv_padding='same',
                       pool_padding='valid', arf=0.2, pool_size=2, dil_rate=1, inputs=inputs)
    modelB = MSM(filters=16, kernerl_size=3, strides=1, conv_padding='same',
                       pool_padding='valid', arf=0.2, pool_size=2, dil_rate=2, inputs=inputs)
    modelC = MSM(filters=16, kernerl_size=3, strides=1, conv_padding='same',
                       pool_padding='valid', arf=0.2, pool_size=2, dil_rate=3, inputs=inputs)
    combined = concatenate([modelA.output, modelB.output, modelC.output])
    z = DCM(filters=64, kernerl_size=3, strides=1, conv_padding='same',
              pool_padding='valid', arf=0.3, pool_size=2, BatchNormal=BatchNorm, inputs=combined)
    z = DCM(filters=32, kernerl_size=3, strides=1, conv_padding='valid',
              pool_padding='valid', arf=0.2, pool_size=2, BatchNormal=BatchNorm, inputs=z)

    z =GlobalMaxPool1D()(z)
    z=Dropout(0.5)(z)
    z = Dense(units=num_classes, activation='softmax')(z)
    model = Model(inputs=inputs, outputs=z)

    # 定义优化器
    Nadam1 = Nadam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    # 定义优化器，loss function, 训练过程中计算准确率
    model.compile(optimizer=Nadam1, loss='categorical_crossentropy', metrics=['acc'])

    # 画出网络结构
#    plot_model(model, to_file='model_cnn.png', show_shapes=True, show_layer_names='False', rankdir='TB')


    callback_list = [ModelCheckpoint(filepath='MSMCNN.hdf5', verbose=1, save_best_only=True, monitor='val_loss')]

    # 训练模型
    model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_data=(x_valid, y_valid), shuffle=True,
              callbacks=callback_list)

    model.load_weights('MSMCNN.hdf5')
    # 评估模型
    loss, accuracy = model.evaluate(x_test, y_test)
    print("test loss:", loss)
    print("test accuracy", accuracy)
    # plot_model(model=model, to_file='mycnn.png', show_shapes=True)

    # 保存模型
    model.save('model_save.h5')

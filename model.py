import csv
from keras.layers.core import Flatten, Dense, Activation, Dropout
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.layers import Cropping2D
from keras import backend as K
from keras.layers import Lambda
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import next_batch


def read_lines():
    """
    读取图像和数据，放入到列表
    :return: 
    """
    samples = []
    with open('./data/driving_log.csv') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            if i > 0:
                samples.append(line)
    return samples


# 建立模型，训练
def create_model_architecture():
    """
    
    创建训练模型,参考NVIDA
    :return: 
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((60, 20), (0, 0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.summary()
    return model


def train_model(model, train_generator, validation_generator):
    """
    训练模型
    :param model: 
    :param train_generator: 
    :param validation_generator: 
    :return: 
    """
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, steps_per_epoch=len(train_samples) / 64, validation_data=validation_generator,
                        validation_steps=len(validation_samples) / 64, nb_epoch=3, verbose=1)


if __name__ == '__main__':
    samples = read_lines()
    # 打乱一下数据
    samples = shuffle(samples)
    # 把数据拆分为训练集和验证集
    train_samples, validation_samples = train_test_split(samples, test_size=0.1)
    # 建立测试集和验证集数据生成器
    train_generator = next_batch(train_samples, batch_size=64)
    validation_generator = next_batch(validation_samples, batch_size=64)
    # 建立模型
    model = create_model_architecture()
    # 训练模型             
    train_model(model, train_generator, validation_generator)
    # 保存训练模型
    model.save('model.h5')
    # 前面导入backend，在模型调用结束时清空一下。
    K.clear_session()

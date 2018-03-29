import csv
from keras.layers.core import Flatten, Dense, Activation
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
from keras.layers import Cropping2D
import numpy as np
import cv2
from keras import backend as K
from keras.layers import Lambda
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


# 读取图像和数据，放入到列表
def read_lines():
    samples = []
    with open('./data/driving_log.csv') as file:
        reader = csv.reader(file)
        for i, line in enumerate(reader):
            if i > 0:
                samples.append(line)
    return samples


# 建立模型，训练
def create_model_architecture():
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0))))
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
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for i, batch_sample in enumerate(batch_samples):
                name = './data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                # trim image to only see section with road
            yield shuffle(np.array(images), np.array(angles))


# read lines and split the samples to train and validation data
samples = read_lines()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# create model
model = create_model_architecture()

# train_generator and validation_generator
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=len(train_samples) / 64, validation_data=validation_generator,
                    validation_steps=len(validation_samples) / 64, nb_epoch=2, verbose=1)

model.save('model.h5')

# 前面导入backend，在模型调用结束时清空一下。
K.clear_session()

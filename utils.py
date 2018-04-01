import numpy as np
import cv2
from sklearn.utils import shuffle


def random_img_flip(img, angle):
    """
    图像水平翻转
    :param img: 
    :param angles: 
    :return: 
    """

    rand = np.random.rand()
    if rand > 0.5:
        img = cv2.flip(img, 1)
        angle = -angle
    return img, angle


def random_img_choose(sample):
    """
    加载图像，可能是左，中，右其中一个
    :return: 
    """
    choice = np.random.choice(3, 1)
    if choice == 0:
        name = './data/IMG/' + sample[0].split("/")[-1]
        center_image = cv2.imread(name)
        center_angle = float(sample[3])
    elif choice == 1:
        name = './data/IMG/' + sample[1].split("/")[-1]
        center_image = cv2.imread(name)
        center_angle = float(sample[3]) + 0.2
    else:
        name = './data/IMG/' + sample[2].split("/")[-1]
        center_image = cv2.imread(name)
        center_angle = float(sample[3]) - 0.2
    return center_image, center_angle


def next_batch(samples, batch_size=32):
    """
    获取一个批次的数据
    
    :param samples: 
    :param batch_size: 
    :return: 
    """

    shuffle(samples)
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for i, sample in enumerate(batch_samples):
                img, angle = random_img_choose(sample)
                img, angle = random_img_flip(img, angle)
                images.append(img)
                angles.append(angle)
        yield shuffle(np.array(images), np.array(angles))

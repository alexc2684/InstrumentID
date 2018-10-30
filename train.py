import os
import numpy as np
from argparse import ArgumentParser

from vgg_model import VGG

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-e', dest='epochs')
    parser.add_argument('-b', dest='batch_size')
    parser.add_argument('-f', dest='model_name')
    args = parser.parse_args()

    train_X = np.load('data/train_X.npy')
    train_y = np.load('data/train_y.npy')
    val_X = np.load('data/val_X.npy')
    val_y = np.load('data/val_y.npy')
    test_X = np.load('data/test_X.npy')
    test_y = np.load('data/test_y.npy')

    INPUT_SHAPE = train_X[0].shape
    print(train_X)
    print(val_X.shape)
    model = VGG(INPUT_SHAPE,
                args.epochs,
                args.batch_size,
                args.model_name)

    print("Training model")
    model.train(train_X, train_y, val_X, val_y)
    # print("Train running!")
    # model.predict(train_X)
    # print("Testing running!")
    # model.predict(test_X)
    print("Finished training")

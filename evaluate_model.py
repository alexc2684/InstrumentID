import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.model_selection import roc_curve, balanced_accuracy_score

from vgg import VGG

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', dest='testset')
    parser.add_argument('-l', dest='labels')
    parser.add_argument('-m', dest='model')

    args = parser.parse_args()

    frames = np.load(args.testset)
    labels = np.load(args.labels)
    INPUT_SHAPE = frames[0].shape
    model = VGG(INPUT_SHAPE,
                1,
                1,
                args.model)

    predictions = model.predict(frames)
    print(roc_curve(labels, predictions))
    print(balanced_accuracy_score(labels, predictions))

import dsdtools
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from helpers import get_filter_bank

SAMPLE_LENGTH = 44100
train_X = []
train_y = []
test_X = []
test_y = []
#function to process each individual track
def process_train_track(track):
    global train_X
    global train_y
    instruments = track.targets
    for instrument in instruments.keys():
        audio = instruments[instrument].audio
        i = 0
        while i + SAMPLE_LENGTH < audio.shape[0]:
            sample = audio[i:i + SAMPLE_LENGTH, 0] #TODO: find out if both channels needed
            banks = get_filter_bank(sample)
            banks = banks.reshape(banks.shape[0], banks.shape[1], 1)
            if instrument == 'vocals':
                train_X.append(banks)
                train_y.append(0)
            elif instrument == 'drums':
                train_X.append(banks)
                train_y.append(1)
            elif instrument == 'bass':
                train_X.append(banks)
                train_y.append(2)

            i += SAMPLE_LENGTH

def process_test_track(track):
    global test_X
    global test_y
    instruments = track.targets
    for instrument in instruments.keys():
        audio = instruments[instrument].audio
        i = 0
        while i + SAMPLE_LENGTH < audio.shape[0]:
            sample = audio[i:i + SAMPLE_LENGTH, 0] #TODO: find out if both channels needed
            banks = get_filter_bank(sample)
            banks = banks.reshape(banks.shape[0], banks.shape[1], 1)
            if instrument == 'vocals':
                test_X.append(banks)
                test_y.append(0)
            elif instrument == 'drums':
                test_X.append(banks)
                test_y.append(1)
            elif instrument == 'bass':
                test_X.append(banks)
                test_y.append(2)

            i += SAMPLE_LENGTH
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', dest='PATH')
    args = parser.parse_args()

    dsd = dsdtools.DB(root_dir=args.PATH)
    dsd.run(process_train_track, subsets='Dev')

    print("Processing training data")
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=.2, random_state=6)
    train_X = np.array(train_X)
    val_X = np.array(val_X)

    np.save('data/train_X.npy', train_X)
    np.save('data/train_y.npy', train_y)
    np.save('data/val_X.npy', val_X)
    np.save('data/val_y.npy', val_y)
    print("Saved training data")

    print("Processing test data")
    dsd.run(process_test_track, subsets='Test')
    test_X, _, test_y, _ = train_test_split(test_X, test_y, test_size=0, random_state=6)
    np.save('data/test_X.npy', test_X)
    np.save('data/test_y.npy', test_y)
    print("Saved test data")

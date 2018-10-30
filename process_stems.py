import musdb
import numpy as np
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from helpers import get_filter_bank

SAMPLE_LENGTH = 44100

def process_tracks(data):
    test = data[0].targets['bass'].audio
    X = []
    y = []

    for track in data:
        instruments = track.targets
        for instrument in instruments.keys():
            audio = instruments[instrument].audio
            for i in range(0, audio.shape[0], SAMPLE_LENGTH):
                sample = audio[i:i + SAMPLE_LENGTH, 0] #TODO: find out if both channels needed
                banks = get_filter_bank(sample)
                banks = banks.reshape(banks.shape[0], banks.shape[1], 1)
                if instrument == 'vocals':
                    X.append(banks)
                    y.append(0)
                elif instrument == 'drums':
                    X.append(banks)
                    y.append(1)
                elif instrument == 'bass':
                    X.append(banks)
                    y.append(2)
    X = np.array(X)
    y = np.array(y)
    return X, y

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', dest='PATH')
    args = parser.parse_args()

    mus = musdb.DB(root_dir=args.PATH)
    train = mus.load_mus_tracks(subsets=['train'])
    test = mus.load_mus_tracks(subsets=['test'])

    print("Processing training data")
    X, y = process_tracks(train)
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=.2, random_state=6)
    train_X = np.array(train_X)
    val_X = np.array(val_X)
    
    np.save('data/train_X.npy', train_X)
    np.save('data/train_y.npy', train_y)
    np.save('data/val_X.npy', val_X)
    np.save('data/val_y.npy', val_y)
    print("Saved training data")

    print("Processing test data")
    test_X, test_y = process_tracks(test)
    test_X, _, test_y, _ = train_test_split(test_X, test_y, test_size=0, random_state=6)
    np.save('data/test_X.npy', test_X)
    np.save('data/test_y.npy', test_y)
    print("Saved test data")

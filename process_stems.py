import musdb
from helpers import get_filter_bank

SAMPLE_LENGTH = 400000
mus = musdb.DB(root_dir='MUS-STEMS-SAMPLE/')

train = mus.load_mus_tracks(subsets=['train'])
test = mus.load_mus_tracks(subsets=['test'])

vocals = []
drums = []
bass = []

def process_tracks(data):
    for track in data:
        instruments = track.targets
        for instrument in instruments.keys():
            audio = instruments[instrument].audio
            for i in range(0, audio.shape[0], SAMPLE_LENGTH):
                sample = audio[i:i + SAMPLE_LENGTH, 0] #TODO: find out if both channels needed
                banks = get_filter_bank(sample)
                if instrument == 'vocals':
                    vocals.append(banks)
                elif instrument == 'drums':
                    drums.append(banks)
                elif instrument == 'bass':
                    bass.append(banks)

process_tracks(train)
print(vocals)

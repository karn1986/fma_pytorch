"""
@author: KAgarwal
Sept 13, 2021
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import os.path
import ast
import torch
import torchvision.transforms.functional as TF
import librosa
from sklearn.preprocessing import LabelEncoder

def load_metadata(filepath):
    '''
    This function is borrowed from the utils.py of the fma repo
    https://github.com/mdeff/fma
    '''
    filename = os.path.basename(filepath)

    if 'features' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filepath, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        try:
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                    'category', categories=SUBSETS, ordered=True)
        except (ValueError, TypeError):
            # the categories and ordered arguments were removed in pandas 0.25
            tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                     pd.CategoricalDtype(categories=SUBSETS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype('category')

        return tracks


def get_audio_path(audio_dir, track_id):
    """
    This function is borrowed from the utils.py of the fma repo
    https://github.com/mdeff/fma
    Return the path to the mp3 given the directory where the audio is stored
    and the track ID.

    Examples
    --------
    >>> import utils
    >>> AUDIO_DIR = os.environ.get('AUDIO_DIR')
    >>> utils.get_audio_path(AUDIO_DIR, 2)
    '../data/fma_small/000/000002.mp3'

    """
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


class audio_clips(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MP3 audio clips in the 
    fma_large dataset
    """
    def __init__(
        self,
        audio_dir = os.path.abspath("../../fma_large"),
        meta_dir = os.path.abspath("../../fma_metadata"),
        transform = None,
        mode = 'train',
        enc = None
    ):
        """
        Args:
            audio_dir: path to the directory containing the mp3 files
            meta_dir: path to the directory containing the audio metadata
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. 
            mode: train or val or test
            enc: Scikitlearn LabelEncoder() for encoding the genres into integers    
        """
        self.audio_dir = audio_dir
        self.meta_dir = meta_dir
        self.transform = transform
        self.enc = enc # scikit learn labelencoder
        tid, target = self._retrieve_metadata(mode)
        assert len(tid) == len(target)
        self.examples = [(tid[i], target[i]) for i in range(len(tid))]
  

    def _retrieve_metadata(self, mode):
        path = self.meta_dir + '/tracks.csv'
        tracks = load_metadata(path)
        
        train = tracks['set', 'split'] == 'training'
        val = tracks['set', 'split'] == 'validation'
        test = tracks['set', 'split'] == 'test'
        
        corrupt_tracks = [98565,98567,98569,99134,108925,133297,1486,5574,
                          65753,80391,98558,98559,98560,98565,98566,98567,98568,
                          98569,98571,99134,105247,108924,108925,126981,127336,
                          133297,143992,2624,3284,8669,10116,11583,12838,13529,
                          14116,14180,20814,22554,23429,23430,23431,25173,25174,
                          25175,25176,25180,29345,29346,29352,29356,33411,33413,
                          33414,33417,33418,33419,33425,35725,39363,41745,42986,
                          43753,50594,50782,53668,54569,54582,61480,61822,63422,
                          63997,72656,72980,73510,80553,82699,84503,84504,84522,
                          84524,86656,86659,86661,86664,87057,90244,90245,90247,
                          90248,90250,90252,90253,90442,90445,91206,92479,94052,
                          94234,95253,96203,96207,96210,98105,98562,101265,101272,
                          101275,102241,102243,102247,102249,102289,106409,106412,
                          106415,106628,108920,109266,110236,115610,117441,127928,
                          129207,129800,130328,130748,130751,131545,133641,133647,
                          134887,140449,140450,140451,140452,140453,140454,140455,
                          140456,140457,140458,140459,140460,140461,140462,140463,
                          140464,140465,140466,140467,140468,140469,140470,140471,
                          140472,142614,144518,144619,145056,146056,147419,147424,
                          148786,148787,148788,148789,148790,148791,148792,148793,
                          148794,148795,151920,155051]
        
        good = ~tracks.index.isin(corrupt_tracks)
        duration = tracks['track', 'duration'] > 29
        has_label = tracks['track', 'genre_top'].notna()
        
        train = good & train & duration & has_label
        val = good & val & duration & has_label
        test = good & test & duration & has_label
        
        labels = tracks['track', 'genre_top']
        if self.enc is None:
            self.enc = LabelEncoder().fit(labels[train])
            
        enc = self.enc
        if mode == 'train':
            tid = tracks.index[train]
            target = enc.transform(labels[train]).astype(np.int64)
        elif mode == 'val':
            tid = tracks.index[val]
            target = enc.transform(labels[val]).astype(np.int64)
        elif mode == 'test':
            tid = tracks.index[test]
            target = enc.transform(labels[test]).astype(np.int64)

        return tid, target

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        tid, target = self.examples[i]
        filename = get_audio_path(self.audio_dir, tid)
        pkl_file = Path(filename[:-3] + 'pkl')
        
        if pkl_file.exists():
            with open(pkl_file, "rb") as f:
                cache = pickle.load(f)
        else:
            cache = {}
        # the slowest operation is decoding the mp3 file
        # Try loading at the native sampling rate first
        if cache.get(tid) is None:
            try:
                x, sr = librosa.load(filename, sr = None, 
                                     mono=True, duration = 29.5)
            except:
                print('track id', tid, ' is corrupt')
                raise
            # if native sr < 22050 load a resampled version
            if sr < 22050:
                x, sr = librosa.load(filename, mono=True, 
                                     duration = 29.5)
            # Limit the number of frames to 128
            hop_length = int(sr * 29.5/128) + 1
            # set STFT window to twice the hop length 
            nfft = hop_length*2
            # The resulting spectrogram should have 128 mel frequencies 
            # and 128 frames
            img = librosa.feature.melspectrogram(x, sr = sr, n_fft=nfft,
                                               hop_length=hop_length,
                                               fmax = 11025)
    
            # Rescale the power to decibel scale    
            img = librosa.power_to_db(img)
            
            cache[tid] = img
            with open(pkl_file, "wb") as f:
                pickle.dump(cache, f)
        else:
            img = cache[tid]
        # if img.shape[1] != 128:
        #     print('sr = ', sr)
        #     print('track id =', tid)
        #     print('track length = ', x.size)
        #     raise
        # Put in form of C x H x W
        img = np.expand_dims(img, axis=0)
        # convert to tensor
        img = torch.from_numpy(img)
        # resize all images to 128 x 128 
        if img.size(-1) != 128:
            img = TF.resize(img, (128,128))
        # Apply any remaaing user specified transforms
        if self.transform is not None:
           img = self.transform(img)
        return (img, target)

# Deep Learning on FMA Dataset using PyTorch

[FMA Website](https://github.com/mdeff/fma) |
[Dataset](https://github.com/mdeff/fma#data) |
[References](#references)

This repository contains PyTorch implementation of a CNN model for music genre classification of the 
[FMA dataset](https://github.com/mdeff/fma). Specifically the `large` subset of data was used and only tracks 
having the top genre label were considered. 

The `audio_clips` is a PyTorch `Dataset` class that reads one track at a time. It takes the Mel spectrogram of the raw
waveform and converts it to a 128 by 128 image. This image is then the input to the CNN model. Since decoding the mp3 files 
is very slow, it dumps the generated spectrogram image into a pickle file for later use. The mp3 decoding is only done once, on subsequent 
fetches if the pickle file is availale, the image is directly read from there.

## Usage

Helpful to get familiar with the dataset by going through some of the jupyter notebooks [here](https://github.com/mdeff/fma#code)

Also look at `usage.py` for reading a track and generating a mel-spectrogram

The `data` module contains `fma_data.py` which contains the class definition of `audio_clips`
The `models` module contains a sample CNN model which operates on 128 x 128 spectrogram images 
`main.py` is the script to train the model
`test_mymodel.py` is to test the trained model on the test set

## Dependencies

Install PyTorch according to the directions at the
[PyTorch Website](https://pytorch.org/get-started/) for your operating system
and CUDA setup. 

In addition following libraries are needed -
* [Librosa](https://github.com/librosa/librosa)

## References

1. Defferrard,  Michäel,  Kirell  Benzi,  Pierre  Vandergheynst,  and  Xavier  Bresson  (2017).  [FMA:  ADataset  for  Music  Analysis](https://arxiv.org/abs/1612.01840).  In:18th International Society for Music Information RetrievalConference (ISMIR). arXiv:1612.01840. 
2. Defferrard, Michäel, Sharada P. Mohanty, Sean F. Carroll, and Marcel Salath́e (2018). [Learning to Recognize Musical Genre from Audio. Challenge Overview](https://arxiv.org/abs/1803.05337). In:The 2018 Web ConferenceCompanion. ACM Press.isbn: 9781450356404.doi:10.1145/3184558.3192310. arXiv:1803.05337. 
3. Dieleman,  Sander  and  Benjamin  Schrauwen  (2014).  “End-to-end  learning  for  music  audio”.  In:2014 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),pp. 6964–6968.doi:10.1109/ICASSP.2014.6854950.
import numpy as np
import wave
import tarfile
import os
from random import randint
import struct


def set_dataset_shape():
    global DATASET_SHAPE
    DATASET_SHAPE = (len(os.listdir(os.path.join("Audio", "Main", "16kHz_16bit"))),10)

set_dataset_shape()

def load_random_batch(batchsize,windowsize,binsize):
    scnorms = [(load_random_scnorm(windowsize,binsize),
                load_random_scnorm(windowsize,binsize)) for i in range(0,batchsize)]
    X = np.array(list(map(scnorm2spec, map(sum_scnorms,scnorms))))
    Y = np.array(list(map(list,map(lambda s: map(scnorm2spec, s), scnorms))))
    return X,Y
    
def load_random_scnorm(windowsize,binsize):
    archindex = randint(0,DATASET_SHAPE[0]-1)
    scnorm = scnorm_from_index(archindex,binsize)
    wstart = randint(0,scnorm.shape[0]-windowsize)
    wend = wstart + windowsize
    return scnorm[wstart:wend]

def scnorm_from_index(index, binsize):
    p = os.path.join("Audio", "Main", "16kHz_16bit")
    archlist = os.listdir(p)
    arch = archlist[index]
    p = os.path.join(p, arch)
    scnorm = tgz2scnorm(p, binsize)
    return scnorm
        

def tgz2scnorm(tarfilename, binsize):
    segments = []
    with tarfile.open(tarfilename, mode="r") as tf:
        audio_names = [n for n in tf.getnames() if n.find(".wav")!=-1]
        audio_names.sort()
        for name in audio_names:
            with tf.extractfile(name) as wf:
                segments.append(wav2scnorm(wf, binsize))
    if(len(segments)==0):
        return np.zeros((1,1))
    return np.concatenate(segments)

def trim_short_files(threshold, binsize):
    i = 0
    while i < DATASET_SHAPE[0]:
        p = os.path.join("Audio", "Main", "16kHz_16bit")
        archlist = os.listdir(p)
        arch = archlist[i]
        p = os.path.join(p, arch)
        scnorm = tgz2scnorm(p, binsize)
        if(scnorm.shape[0]<threshold):
            os.remove(p)
            set_dataset_shape()
        

def scnorm2spec(scnorm):
    np_spec = np.fft.fft(scnorm)
    np_spec_amplitude = np.abs(np_spec)
    np_spec_phase = np.angle(np_spec)
    np_spec_fl = np.stack((np_spec_amplitude,np_spec_phase), axis=2)
    return np_spec_fl

def wav2scnorm(fp, bin_size):
    np_raw_audio = None
    f = wave.open(fp)
    sample_width = f.getsampwidth()
    num_frames = int(f.getnframes())
    num_bins = int(num_frames/bin_size)
    short_slices = []
    for i in range(num_bins):
        raw_audio = f.readframes(bin_size)
        raw_audio_shorts = struct.iter_unpack("h",raw_audio)
        short_slices.append(raw_audio_shorts)
    np_shorts = np.stack(raw_audio_shorts, axis=1)
    np_raw_audio = np_shorts/32768.0
    return np_raw_audio  
        
def downsample(scnorm):
    return np.delete(scnorm, list(range(0,scnorm.shape[1], axis=1)))
    
def sum_scnorms(scnorms):
    return sum(map(lambda x: x/len(scnorms), scnorms))


if (__name__=='__main__'):
    trim_short_files(1500,256)

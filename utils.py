import numpy as np
import pandas as pd
from scipy.signal import welch

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)


# first strategy
def get_extracts(file_names, targets, sfreqs):
    """
    Args:
    file_names = [fn1.csv, fn2.csv, ...]
    targets = ['trauma', 'healthy', ...]
    sfreqs = [freq1, freq2, ...]
    
    #records = [np.array[n_samples, n_channels], np.array, np.array, ..]
    
    Return:
    records = []
    """
    subjects = []
    records = []
    targets_chunk = []
    for fn, target, sfreq in zip(file_names, targets, sfreqs):
        record = pd.read_csv(f'preproc_data/{fn}')
        if len(record) < 60 * sfreq:
            continue
        records.append(record[:60*sfreq])
        targets_chunk.append(target)
        subjects.append(fn)
        
    return records, targets_chunk, subjects


def plot_rawsignal(time, signal, channel):
    # fig= plt.figure(figsize=(15,4))
    # plt.plot(records[20]['time'], records[20]['t4'])

    # Plot the signal
    fig= plt.figure(figsize=(15,4))
    plt.plot(time, signal, lw=1.5, color='k')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage')
    plt.xlim([time.min(), time.max()])
    plt.title(f'Brain activity, EEG data ({channel})')
    sns.despine()
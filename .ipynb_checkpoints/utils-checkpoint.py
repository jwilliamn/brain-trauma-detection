import numpy as np
import pandas as pd

import mne
from os import mkdir
from os.path import exists, join
import itertools
from functools import partial
from scipy import signal
from scipy.signal import hilbert, welch
from scipy.stats import pearsonr
from scipy.integrate import simps
from tqdm import tqdm
from mne.connectivity import spectral_connectivity, phase_slope_index


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)


# Length of signal homogeneity
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

        # Bad signal removal
        record = record.drop(['cz', 'p4'], axis=1)
        chunk = record[:60*sfreq]
        chunk.to_csv(f'preproc_data/{fn}', index=False)
        
        records.append(chunk)
        targets_chunk.append(target)
        subjects.append(fn)
        
    return records, targets_chunk, subjects


def plot_rawsignal(time, signal, channel):

    # Plot the signal
    fig= plt.figure(figsize=(15,4))
    plt.plot(time, signal, lw=1.5, color='k')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Voltage')
    plt.xlim([time.min(), time.max()])
    plt.title(f'Brain activity, EEG data ({channel})')
    sns.despine()
    
    

    
band_bounds = {'theta' : [4, 8],'alpha': [8, 13],
               'beta': [13, 30],'gamma': [30, 45]}

dict_regions = {'prefrontal':['fp1','fp2'],
                   'frontal':['f7','f3','f4','fz','f8'],
                   'central':['t3','c3','cz','c4','t4'], 
                   'parietal':['t5','p3','pz','p4','t6'],
                   'occipital':['o1','o2']}



# define col name
def get_col_name(method, band, ch_1, ch_2=None):
    band_name = 'nofilt' if band is None else band
    s = method + '_' + band_name + '_' + ch_1
    if ch_2:
        s += '_' + ch_2
    return s

# define file path
def get_feature_path(path, method_name):
    return join(path, method_name + '.csv')

# Filter for each band
def get_filter(sfreq=125., band='alpha'):

    f_low_lb = band_bounds[band][0] - 1
    f_low_ub = band_bounds[band][0]
    f_high_lb = band_bounds[band][1]
    f_high_ub = band_bounds[band][1] + 1

    nyq = sfreq / 2.  # the Nyquist frequency is half our sample rate

    freq = [0., f_low_lb, f_low_ub, f_high_lb, f_high_ub, nyq]
    gain = [0, 0, 1, 1, 0, 0]
    n = int(round(1 * sfreq)) + 1
    filt = signal.firwin2(n, freq, gain, nyq=nyq)
    return filt

# Calculate features according to ratio bands
def get_bands_feats(df, sfreq=125.):

    electrodes = df.columns

    feats = {}

    for el in electrodes:
        freqs, psds = signal.welch(df[el], sfreq, nperseg=1024)
        fres = freqs[1] - freqs[0]
        psd_df = pd.DataFrame(data={'freqs': freqs, 'psds': psds})

        feats[get_col_name('bands', 'alpha', el)] = psd_df.loc[
            (psd_df['freqs'] >= band_bounds['alpha'][0]) &
            (psd_df['freqs'] <= band_bounds['alpha'][1])]['psds'].sum()/psd_df['freqs'].sum()

        feats[get_col_name('bands', 'beta', el)] = psd_df.loc[
            (psd_df['freqs'] >= band_bounds['beta'][0]) &
            (psd_df['freqs'] <= band_bounds['beta'][1])]['psds'].sum()/psd_df['freqs'].sum()
        
        feats[get_col_name('bands', 'theta', el)] = psd_df.loc[
            (psd_df['freqs'] >= band_bounds['theta'][0]) &
            (psd_df['freqs'] <= band_bounds['theta'][1])]['psds'].sum()/psd_df['freqs'].sum()
        
        feats[get_col_name('bands', 'gamma', el)] = psd_df.loc[
            (psd_df['freqs'] >= band_bounds['gamma'][0]) &
            (psd_df['freqs'] <= band_bounds['gamma'][1])]['psds'].sum()/psd_df['freqs'].sum()

    return feats

# Calculate spectral connectivity
def get_mne_spec_con_feats(df, sfreq=125., band=None, method='coh'):

    electrodes = df.columns
    res = spectral_connectivity(
        df[electrodes].values.T.reshape(1, len(electrodes), -1),
        method=method, sfreq=sfreq, verbose=False)

    data = res[0]
    freqs = res[1]

    def filter(arr):
        if band is None:
            return arr
        else:
            start_idx = np.where(freqs > band_bounds[band][0])[0][0]
            end_idx = np.where(freqs < band_bounds[band][1])[0][-1] + 1
            return arr[start_idx:end_idx]

    d = {}

    idx_electrodes_dict = {i: e for i, e in enumerate(electrodes)}

    for idx_1, idx_2 in itertools.combinations(range(len(electrodes)), 2):
        el_1 = idx_electrodes_dict[idx_1]
        el_2 = idx_electrodes_dict[idx_2]
        d[get_col_name(method, band, el_1, el_2)] = filter(data[idx_2, idx_1]).mean()

    return d

def get_envelope_feats(df, sfreq=125., band='alpha'):

    electrodes = df.columns

    df = df.copy()
    new_df = pd.DataFrame()
    if band is not None:
        filt = get_filter(sfreq, band)
    else:
        filt = None

    for el in electrodes:
        sig = df[el]
        if filt is not None:
            sig = np.convolve(filt, df[el], 'valid')
        sig = hilbert(sig)
        sig = np.abs(sig)
        new_df[el + '_env'] = sig

    d = {}

    idx_electrodes_dict = {i: e for i, e in enumerate(electrodes)}

    for idx_1, idx_2 in itertools.combinations(range(len(electrodes)), 2):
        el_1 = idx_electrodes_dict[idx_1]
        el_2 = idx_electrodes_dict[idx_2]
        series_1 = new_df[el_1 + '_env']
        series_2 = new_df[el_2 + '_env']
        d[get_col_name('env', band, el_1, el_2)] = pearsonr(series_1, series_2)[0]

    return d


def genFeatures(method_name, data_path, name_file, out_path):
    
    methods = {
    'coh': partial(get_mne_spec_con_feats, band=None, method='coh'),
    'coh-alpha': partial(get_mne_spec_con_feats, band='alpha', method='coh'),
    'coh-beta': partial(get_mne_spec_con_feats, band='beta', method='coh'),
    'env': partial(get_envelope_feats, band=None),
    'env-alpha': partial(get_envelope_feats, band='alpha'),
    'env-beta': partial(get_envelope_feats, band='beta'),
    'bands': get_bands_feats,
    }

    f = methods[method_name]
    
    def unity_func(x):
        return x

    df_filter_func = unity_func

    path_file_path = join(data_path, name_file)
    path_df = pd.read_csv(path_file_path)
    # required columns check
    assert all([col in path_df.columns for col in ['fn', 'target']])

    features_path = get_feature_path(out_path, method_name)

    new_rows = []

    for i, row in tqdm(path_df.iterrows(), total=len(path_df)):
        try:
            path = join(data_path, row['fn'])
            df = pd.read_csv(path, index_col='time')
            df = df_filter_func(df)
            new_row = f(df)
        except AssertionError:
            print('Error in file ' + row['fn'])
            continue
        except FileNotFoundError:
            print('Not found - ' + row['fn'])
            continue

        for col in ['fn', 'target']:
            new_row[col] = row[col]
        new_rows.append(new_row)

        res_df = pd.DataFrame(new_rows)
        res_df.to_csv(features_path, index=False)
        
    #return res_df
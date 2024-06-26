import os
import wfdb
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.interpolate import CubicHermiteSpline
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.signal import detrend

def read_ctg(record_path):
    record = wfdb.rdrecord(record_path)
    fhr_signal = record.p_signal[:, 0]
    uc_signal = record.p_signal[:, 1]
    
    return fhr_signal, uc_signal

# preprocess the fhr solely
def remove_gap_fhr(signal):
    is_zero = signal == 0
    zero_groups = np.diff(np.where(np.concatenate(([is_zero[0]], is_zero[:-1] != is_zero[1:], [True])))[0])[::2]  # 计算连续0组的长度
    cleaned_signal_segments = []
    index_change = np.where(np.concatenate(([is_zero[0]], is_zero[:-1] != is_zero[1:], [True])))[0][::2]
    last_nonzero_index = 0
    for start, length in zip(index_change, zero_groups):
        if length > 60:
            cleaned_signal_segments.append(signal[last_nonzero_index:start])
            last_nonzero_index = start+length
        else:
            continue
    cleaned_signal_segments.append(signal[last_nonzero_index:])
    cleaned_signal = np.concatenate(cleaned_signal_segments)
    
    return cleaned_signal

def hermite_interpolation(signal):
    timestamps = np.arange(len(signal))
    # 识别非零数据点
    non_zero_indices = np.where(signal != 0)[0]
    non_zero_fhr = signal[non_zero_indices]
    # 对非零部分进行插值
    dydx = np.diff(non_zero_fhr) / np.diff(non_zero_indices) #一阶导数
    dydx = np.append(dydx, dydx[-1])

    hermite_spline = CubicHermiteSpline(non_zero_indices, non_zero_fhr, dydx)
    interpolated_data = hermite_spline(timestamps)
    
    return interpolated_data


def linear_interpolation(signal):
    interpolated_signal = np.copy(signal)
    non_zero_indices = np.nonzero(signal)[0]
    
    for i in range(len(non_zero_indices) - 1):
        start_index = non_zero_indices[i]
        end_index = non_zero_indices[i + 1]
        if end_index - start_index > 1:
            start_value = signal[start_index]
            end_value = signal[end_index]
            slope = (end_value - start_value) / (end_index - start_index)
            for j in range(1, end_index - start_index):
                interpolated_signal[start_index + j] = start_value + slope * j
                
    num_zero = len(signal) - len(non_zero_indices)
    
    return interpolated_signal, num_zero



# normal fhr range from 120 to 160 bpm
def outlier_detection_fhr(signal):
    # isolation forest选出的outliner
    '''
    iso = IsolationForest(contamination=0.01)
    outliers_iso = iso.fit_predict(signal.reshape(-1, 1))
    outliers = signal[outliers_iso == -1] 
    '''
    # mannualy selecte outliner, which are above 160bpm or below 60 bpm 
    outliers = signal[(signal >= 200) | (signal<=60)]
    outlier_indices = np.where((signal >= 200) | (signal <= 60))
    timestamps = np.arange(len(signal))
    timestamps_clean = np.delete(timestamps, outlier_indices)
    data_clean = np.delete(signal, outlier_indices)

    dydx = np.gradient(data_clean, timestamps_clean)

    hermite_spline = CubicHermiteSpline(timestamps_clean, data_clean, dydx)
    data_interpolated = hermite_spline(timestamps)
    
    return data_interpolated
    
def detrend_fhr(signal):
    detrended_data = detrend(signal) + np.mean(signal)
    return detrended_data


def preprocessing_fhr(data):
    gap_removed_data = remove_gap_fhr(data)
    interpolated_data, num_zero = linear_interpolation(gap_removed_data)
    detrended_data = detrend_fhr(interpolated_data)
    #outlier_removed_data = outlier_detection_fhr(detrended_data)
    preprocessed_fhr = detrended_data
    
    return preprocessed_fhr, num_zero

def segment_fhr(data_path, length, save_dir):
    # 20 minutes, 1200 seconds, 4800 points
    basename = os.path.basename(data_path)
    fhr_signal, uc_signal = read_ctg(data_path)
    preprocessed_fhr, num_zero = preprocessing_fhr(fhr_signal)
    n_fragments = math.ceil(len(preprocessed_fhr) / length)
    for i in range(n_fragments):
        if i!= n_fragments - 1:
            fragment = preprocessed_fhr[i*4800: (i+1)*4800]
        else:
            fragment = preprocessed_fhr[-4800:]
            
        if num_zero/len(fhr_signal) < 0.1:
            save_path = os.path.join(save_dir, basename+'_'+str(i)+'.npy')
            np.save(save_path, fragment)
        


files_dir = '../CTU-CHB-original'
basenames = []
for file in os.listdir(files_dir):
    if os.path.basename(file)[:-4] not in basenames:
        if len(os.path.basename(file)[:-4]) == 4:
            basenames.append(os.path.basename(file)[:-4])
        
for i in range(len(basenames)):
    record_path = os.path.join(files_dir, basenames[i])
    segment_fhr(data_path=record_path, length=4800, save_dir='../2-classification')
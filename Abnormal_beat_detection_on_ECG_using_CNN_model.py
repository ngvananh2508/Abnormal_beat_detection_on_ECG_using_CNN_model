# %% 
import os
import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout


# %%
# Get the list of all patients ID in the data
def extract_numbers_from_filenames(folder_path):
    numbers = set()
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            if file.endswith(('.atr', '.dat', '.hea', '.xws')):
                number = file.split('.')[0]
                if number.isdigit():
                    numbers.add(number)
    return sorted(list(numbers))

folder_path = '/home/nguyen-van-anh/Desktop/MIT_BH_data/physionet.org/files/mitdb/1.0.0/'
number_list = extract_numbers_from_filenames(folder_path)
print(len(number_list))


# %%
# Check the first patient
pt = '100'
file = folder_path + pt
record = wfdb.rdrecord(file)
print(record.p_signal.shape)
print(record.fs)

annot = wfdb.rdann(file, 'atr')
print(annot.symbol)
print(annot.sample)


# %%
df = pd.DataFrame()
for pt in number_list:
    file = folder_path + pt
    annotation = wfdb.rdann(file, 'atr')
    sym = annotation.symbol
    unique_symbols , counts = np.unique(sym, return_counts=True)
    df_sub = pd.DataFrame({'symbols': unique_symbols, 'counts': counts, 'patients': [pt]*len(counts)})
    df = pd.concat([df, df_sub], axis = 0)
print(df)

# %%
# Symbols for normal and abnormal beats
abnormal = ['L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r', 'F', 'e', 'j', 'n', 'E']
normal = ['N']

def load_ecg(file):
    record = wfdb.rdrecord(file)
    annotation = wfdb.rdann(file, 'atr')
    p_signal = record.p_signal

    assert record.fs == 360, 'sampling freq is not 360'

    symbols = annotation.symbol
    symbol_samples = annotation.sample

    return p_signal, symbols, symbol_samples

# Extract fixed-length segments of an ECG signal around annotated beats
def build_XY(p_signal, df_ann, num_cols, abnormal, num_sec, fs): #num_cols: length of each ECG segment
    num_rows = len(df_ann)
    X = np.zeros((num_rows, num_cols))
    Y = np.zeros((num_rows, 1))
    sym = []
    max_row = 0
    for atr_sample, atr_sym in zip(df_ann.atr_sample.values, df_ann.atr_sym.values):
        left = max([0, (atr_sample - num_sec*fs)])
        right = min([len(p_signal), (atr_sample + num_sec*fs)])
        x = p_signal[left:right]
        if len(x) == num_cols:
            X[max_row, :] = x
            Y[max_row, :] = int(atr_sym in abnormal)
            sym.append(atr_sym)
            max_row += 1
    X = X[:max_row, :]
    Y = Y[:max_row, :]
    return X, Y, sym

def make_dataset(pts, num_sec, fs, abnormal):
    num_cols = 2*num_sec*fs
    X_all = np.zeros((1, num_cols))
    Y_all = np.zeros((1, 1))
    sym_all = []
    max_rows = []
    for pt in pts:
        file = folder_path + pt
        p_signal, atr_sym, atr_sample = load_ecg(file) #atr_sample: location of this beat
        p_signal = p_signal[:, 0]

        df_ann = pd.DataFrame({'atr_sym': atr_sym, 'atr_sample': atr_sample})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(abnormal + ['N'])]

        X, Y, sym = build_XY(p_signal, df_ann, num_cols, abnormal, num_sec, fs)
        sym_all = sym_all + sym
        max_rows.append(X.shape[0])
        X_all = np.append(X_all, X, axis = 0)
        Y_all = np.append(Y_all, Y, axis = 0)
    
    X_all = X_all[1:, :]
    Y_all = Y_all[1:, :]

    assert np.sum(max_rows) == X_all.shape[0], 'number of X, max_rows rows messed up'
    assert Y_all.shape[0] == X_all.shape[0], 'number of X, Y rows messed up'
    assert Y_all.shape[0] == len(sym_all), 'number of Y, sym rows messed up'
    return X_all, Y_all, sym_all


# %%
num_sec = 3
fs = 360
X_all, Y_all, sym_all = make_dataset(number_list, num_sec, fs, abnormal)
print(X_all.shape)
print(Y_all.shape)
print(len(sym_all))


# %%
# Check the shape of wave in the ECG data
p_signal, symbols, symbol_samples = load_ecg(file)
values, counts = np.unique(symbols, return_counts=True)
abnormal_idx = [b for a, b in zip(symbols, symbol_samples) if a in abnormal]
normal_idx = [b for a, b in zip(symbols, symbol_samples) if a in normal]

segment_len = 6
fs = 360

segment_start = abnormal_idx[0] - int(segment_len/2*fs)
segment_end = abnormal_idx[0] + int(segment_len/2*fs)

x = np.arange(len(p_signal))


plt.figure()
plt.plot(x, p_signal[:, 0])
plt.plot(x[normal_idx], p_signal[normal_idx, 0], 'b*', label = 'normal')
plt.plot(x[abnormal_idx], p_signal[abnormal_idx, 0], 'ro', label = 'abnormal')
plt.xlim(segment_start, segment_end)


# %%
X_train, X_test, y_train, y_test = train_test_split(X_all, Y_all, test_size=0.2, stratify=Y_all, random_state=12)


# %%
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# %%
model = Sequential()
model.add(Conv1D(128, 5, padding='same', activation='relu', input_shape=(2160,1)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train_cnn, y_train, batch_size=32, epochs=3, verbose=1)


# %%
loss, acc = model.evaluate(X_test_cnn, y_test)
print(acc)
# %%

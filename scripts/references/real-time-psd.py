from htil_eeg import HTIL_EEG 
from brainflow import BoardIds
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path 
import mne
import time
import math
import os
import random
import numpy as np
import winsound
duration = 1000  # milliseconds
freq = 440  # Hz

h_eeg = HTIL_EEG(BoardIds.CROWN_BOARD)

def play_sound(freq, duration):
    winsound.Beep(freq, duration)


def raw_to_epoch_dataframe(data, label, stop):
    processed_data = h_eeg.preprocess_data(data, show_muscle_artifacts=True, show_plot=True)
    epochs = h_eeg.get_epochs(processed_data, label, 1, start=1.0, stop=stop, duration=1.0, show_log=False, drop_bad=True, plot_epochs=True)
    df = h_eeg.epochs_to_dataframe(epochs, label, freq_max=40)
    return df

def get_data(data_length, label, stop):
    df, processed, raw = h_eeg.get_crown_data(data_length)
    df = raw_to_epoch_dataframe(processed, label, stop)
    return df

def main():
    lenght = 30
    print("Start right foot condition.")
    h_eeg.start_stream()
    df_cond_a = get_data(lenght, "right", lenght)
    k=input("Press enter to start next recording") 
    print("Start left foot condition.")
    df_cond_b = get_data(lenght, "left", lenght)
    combined_df = pd.concat([df_cond_a, df_cond_b], ignore_index=True)
    df_C3 = combined_df[combined_df['Channel'].isin(['C3'])]
    plt.clf()
    sns.lineplot(data=df_C3, x="Frequency", y="Value", hue="Label").set(title='C3')
    plt.savefig("imgs/T1.png")

def create_event(start_time, label, sfreq):
    elapsed = math.floor((time.time() - start_time) * sfreq)
    return [elapsed, 0, label]

def create_event_array(event_labels, event_frequency, total_epochs, shuffle=True):
    events = []
    for index, label in enumerate(event_labels):
        freq = round(total_epochs * event_frequency[index])
        for j in range(freq):
            events.append(label)
        #print(i, event_labels[i], freq)
        #print(i, event_frequency[i]) 
        #label = event_labels[i]
        #events.append(create_event(start_time, label, 256))
        #time.sleep(event_frequency)
    if shuffle:
        random.shuffle(events)
    return events

def run_experiment(experiment_name, total_epochs=10):
    events = []
    k=input("Start experiment")
    epoch_length = 2
    event_dict = {"rest": 1, "clench": 2}
    event_labels = create_event_array(["rest", "clench"], [0.5, 0.5], total_epochs)
    h_eeg.start_stream()
    start_time = time.time()
    for event in event_labels:
        print(f"stimulus {event}")
        time.sleep(epoch_length)
        events.append(create_event(start_time, event_dict[event], 256))
    df, processed, raw = h_eeg.get_recent_crown_data()
    np_events = np.array(events)
    epochs = mne.Epochs(processed, np_events, tmin=0, tmax=1.0, event_id=event_dict, baseline=(0, 0))
    epochs.plot(event_id=True, events=True, event_color={1:"blue", 2:"red"}, scalings=dict(eeg='1e-4', emg='1e-4'))

    # Save Files
    df.to_csv(f'data/csv/{experiment_name}.csv', index=False)   
    raw.save(Path(f'data/raw/{experiment_name}_raw.fif'), overwrite=True)
    epochs.save(Path('data/epochs') / f'{experiment_name}_epo.fif', overwrite=True)
    np.savetxt(f"data/events/{experiment_name}_events.csv", np_events, delimiter=',', fmt='%d')
    k=input("Press enter to continue.")

def run_experiment_mi(experiment_name, total_epochs=2, epoch_length=2):
    events = []
    k=input("Start experiment")
    #epoch_length = 10
    event_dict = {"right": 1, "left": 2}
    event_labels = create_event_array(["right", "left"], [0.5, 0.5], total_epochs, shuffle=True)
    h_eeg.start_stream()
    start_time = time.time()
    for event in event_labels:
        print(f"______________")
        print(f"stimulus {event}")
        play_sound(freq, duration)
        time.sleep(epoch_length)
        events.append(create_event(start_time, event_dict[event], 256))
    df, processed, raw = h_eeg.get_recent_crown_data()
    np_events = np.array(events)
    epochs = mne.Epochs(processed, np_events, tmin=-1.0, tmax=1.0, event_id=event_dict, baseline=(-0.5, 0))
    epochs.plot(event_id=True, events=True, event_color={1:"blue", 2:"red"}, scalings=dict(eeg='1e-4', emg='1e-4'))

    # Save Files
    df.to_csv(f'data/csv/{experiment_name}.csv', index=False)   
    raw.save(Path(f'data/raw/{experiment_name}_raw.fif'), overwrite=True)
    epochs.save(Path('data/epochs') / f'{experiment_name}_epo.fif', overwrite=True)
    np.savetxt(f"data/events/{experiment_name}_events.csv", np_events, delimiter=',', fmt='%d')
    k=input("Press enter to continue.")

def run_experiment_eyes(experiment_name, total_epochs=2, epoch_length=1):
    events = []
    k=input("Start experiment")
    #epoch_length = 10
    event_dict = {"open": 1, "closed": 2}
    event_labels = create_event_array(["open", "closed"], [0.5, 0.5], total_epochs, shuffle=True)
    h_eeg.start_stream()
    start_time = time.time()
    for event in event_labels:
        print(f"______________")
        print(f"stimulus {event}")
        play_sound(freq, duration)
        time.sleep(epoch_length)
        events.append(create_event(start_time, event_dict[event], 256))
    df, processed, raw = h_eeg.get_recent_crown_data()
    np_events = np.array(events)
    epochs = mne.Epochs(processed, np_events, tmin=-1.0, tmax=1.0, event_id=event_dict, baseline=(-0.5, 0))
    #epochs.plot(event_id=True, events=True, event_color={1:"blue", 2:"red"}, scalings=dict(eeg='1e-4', emg='1e-4'))

    # Save Files
    df.to_csv(f'data/csv/{experiment_name}.csv', index=False)   
    raw.save(Path(f'data/raw/{experiment_name}_raw.fif'), overwrite=True)
    epochs.save(Path('data/epochs') / f'{experiment_name}_epo.fif', overwrite=True)
    np.savetxt(f"data/events/{experiment_name}_events.csv", np_events, delimiter=',', fmt='%d')
    k=input("Press enter to continue.")


def run_single_condition_experiment(experiment_name, total_epochs=2, epoch_length=1):
    events = []
    k=input("Start experiment")
    #epoch_length = 10
    event_dict = {"closed": 1}
    event_labels = create_event_array(["closed"], [1.0], total_epochs, shuffle=True)
    h_eeg.start_stream()
    start_time = time.time()
    for event in event_labels:
        print(f"______________")
        print(f"stimulus {event}")
        time.sleep(epoch_length)
        events.append(create_event(start_time, event_dict[event], 256))
    play_sound(freq, duration)
    df, processed, raw = h_eeg.get_recent_crown_data()
    np_events = np.array(events)
    epochs = mne.Epochs(processed, np_events, tmin=-1.0, tmax=1.0, event_id=event_dict, baseline=(-0.5, 0))
    #epochs.plot(event_id=True, events=True, event_color={1:"blue", 2:"red"}, scalings=dict(eeg='1e-4', emg='1e-4'))

    freqs = np.logspace(*np.log10([1, 20]), num=20)
    n_cycles = freqs / 2.0
    baseline = (None, 0)
    epochs.compute_psd(fmax=30).plot(picks=["C3","C4", "F5", "F6"])
    epochs_tfr = epochs.compute_tfr("morlet", n_cycles=n_cycles, return_itc=False, freqs=freqs, average=True, use_fft=True)
    epochs_tfr.crop(-0.1, 0.7)
    epochs_tfr.plot(title="auto", picks=["C3", "C4", "F5", "F6"], baseline=baseline, mode="logratio") # vmax=1e-8,
    target_freq = 11
    epochs_tfr.plot_joint(timefreqs=((0.15, target_freq), (0.5, target_freq), (0.6, target_freq)),  baseline=baseline)

    # Save Files
    df.to_csv(f'data/csv/{experiment_name}.csv', index=False)   
    raw.save(Path(f'data/raw/{experiment_name}_raw.fif'), overwrite=True)
    epochs.save(Path('data/epochs') / f'{experiment_name}_epo.fif', overwrite=True)
    np.savetxt(f"data/events/{experiment_name}_events.csv", np_events, delimiter=',', fmt='%d')
    k=input("Press enter to continue.")

def load_events():
    events = np.loadtxt("data/events/test1_events.csv", delimiter=',', dtype=int)
    print(events)
    return events

def load_epochs(file_path, event_colors):
    epochs = h_eeg.load_epochs(file_path)
    print(epochs.info)
    print(epochs)
    #epochs.plot(event_id=True, events=True, event_color=event_colors, scalings=dict(eeg='1e-4', emg='1e-4'))
    #k=input("Press enter to continue.")
    return epochs


#run_experiment("test_6", total_epochs=5)
run_experiment_mi("right_left_8", total_epochs=50, epoch_length=2)
#run_experiment_eyes("eyes_open_closed_1", total_epochs=10)
#run_single_condition_experiment("closed_7", total_epochs=20, epoch_length=1)


#epochs = load_epochs('data/epochs/test_2_epo.fif', {1:"blue", 2:"red"})
'''
epochs = h_eeg.load_epochs('data/epochs/test_2_epo.fif')
print(epochs)
rest_epochs = epochs['rest'].average()
print(rest_epochs)
clench_epochs = epochs['clench'].average()
print(clench_epochs)
rest_epochs.plot()
clench_epochs.plot()
k=input("Press enter to continue.")
'''
#run_experiment("test_2")
#main()
'''
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

tmin, tmax = -1.0, 4.0
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw = read_raw_edf(raw_fnames[0], preload=True)
new_events = mne.make_fixed_length_events(raw, start=1, stop=5, duration=1.0)

print(new_events)
'''

'''
epochs_closed = load_epochs('eyes_closed_epo.fif')
epochs_open = load_epochs('eyes_open_epo.fif')
df_closed = h_eeg.epochs_to_dataframe(epochs_closed, "eyes_closed", freq_max=30)
df_open = h_eeg.epochs_to_dataframe(epochs_open, "eyes_open", freq_max=30)

combined_df = pd.concat([df_closed, df_open], ignore_index=True)
cond_a = "F5"
cond_b = "F6"
df_C3 = combined_df[combined_df['Channel'].isin([cond_a])]
df_C4 = combined_df[combined_df['Channel'].isin([cond_b])]

plt.clf()
sns.lineplot(data=df_C3, x="Frequency", y="Value", hue="Label").set(title=cond_a)
plt.savefig(f"imgs/{cond_a}_open_closed.png")
#plt.show()

plt.clf()
sns.lineplot(data=df_C4, x="Frequency", y="Value", hue="Label").set(title=cond_b)
plt.savefig(f"imgs/{cond_b}_open_closed.png")
'''
#print(df_closed)


#print(epochs_closed)
#print(epochs_open)


                              








#h_eeg.save_data(data, 'data/heeg_crown.csv')
'''
lf = h_eeg.csv_to_epoch_dataframe("data/right_foot.csv", "right_foot", 1, show_muscle_artifacts=False, show_log=False, drop_bad=False, show_plot=False)
rf = h_eeg.csv_to_epoch_dataframe("data/left_foot.csv", "left_foot", 1, show_muscle_artifacts=False, show_log=False, drop_bad=False, show_plot=False)
combined_df = pd.concat([lf, rf], ignore_index=True)
df_C3 = combined_df[combined_df['Channel'].isin(['C3'])]
df_C4 = combined_df[combined_df['Channel'].isin(['C4'])]
plt.clf()
sns.lineplot(data=df_C3, x="Frequency", y="Value", hue="Label").set(title='C3')
plt.savefig("imgs/h_eeg.png")
'''
#curry base 
#oscar

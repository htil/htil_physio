import argparse
from brainflow import BoardIds
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import pandas as pd
import numpy as np
import mne
from mne.preprocessing import annotate_muscle_zscore
import matplotlib.pyplot as plt
import seaborn as sns
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                               corrmap)


BOARD = BoardIds.CROWN_BOARD.value
Z_SCORE_THRESHOLD = 1.0

def get_arguments():
    argument_info = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_1', type=str, help='First file to analyze', required=True, default='')
    parser.add_argument('-file_2', type=str, help='Second file to analyze', required=False, default='')

    parser.add_argument('-epochs_start', type=str, help='Time to start epochs (seconds)', required=False, default='1.0')
    parser.add_argument('-epochs_stop', type=str, help='Time to end epochs (seconds)', required=False, default='5.0')
    parser.add_argument('-epochs_duration', type=str, help='Duration of epochs (seconds)', required=False, default='1')
    
    args = parser.parse_args()
    argument_info['file_1'] = args.file_1
    argument_info['file_2'] = args.file_2
    argument_info['epochs_start'] = float(args.epochs_start)
    argument_info['epochs_stop'] = float(args.epochs_stop) 
    argument_info['epochs_duration'] = float(args.epochs_duration)

    return argument_info

def markMuscleArtifacts(raw, threshold, plotLog=True):
    threshold_muscle = threshold  # z-score
    annot_muscle, scores_muscle = annotate_muscle_zscore(
    raw, ch_type="eeg", threshold=threshold_muscle, min_length_good=0.2, filter_freq=[0, 60])
    raw.set_annotations(annot_muscle)

    if plotLog:
        fig, ax = plt.subplots()
        ax.plot(raw.times, scores_muscle)
        ax.axhline(y=threshold_muscle, color='r')
        ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity')
        #fig.savefig('muscle_annot.png')

def preprocess_data(data, sfreq=256, ch_types='eeg', l_freq=1, h_freq=30, show_muscle_artifacts=False, show_plot=False):
    eeg_channels_names = BoardShim.get_eeg_names(BOARD)
    ch_types = ['eeg'] * len(eeg_channels_names)

    # Create MNE info object
    info = mne.create_info(ch_names = eeg_channels_names, sfreq = sfreq, ch_types=ch_types)

    # Create MNE raw object
    raw = mne.io.RawArray(data, info)

    # Detect muscle artifacts
    markMuscleArtifacts(raw, Z_SCORE_THRESHOLD, plotLog=show_muscle_artifacts)

    # Band pass filter
    raw_highpass = raw.copy().filter(l_freq=l_freq, h_freq=h_freq)
    # Convert from uV to V for MNE
    raw_highpass.apply_function(lambda x: x * 1e-6)

    # Format data date for annotations later
    raw_highpass.set_meas_date(0)
    raw_highpass.set_montage("standard_1020")

    # Re-reference the data to average
    eeg_reref_highpass, _ = mne.set_eeg_reference(raw_highpass)

    if show_plot:
        eeg_reref_highpass.plot(clipping=None)
        k=input("press close to exit") 

    return eeg_reref_highpass

def csv_to_dataframe(file):
    eeg_channels_names = BoardShim.get_eeg_names(BOARD)
    df = pd.read_csv(file, usecols = eeg_channels_names).transpose()
    return df

def get_epochs(data, event_label, event_id, start=1, stop=5, duration=1.0, show_log=False, drop_bad=True, plot_epochs=False):
    # Only works for continuous file with only one event type
    events = mne.make_fixed_length_events(data, id=event_id, start=start, stop=stop, duration=duration)
    events_id = {event_label: event_id}
    epochs = mne.Epochs(data, events, tmin=0, tmax=1, event_id=events_id, baseline=(0, 0))
    if drop_bad:
        epochs.drop_bad()
    
    if show_log:
        epochs.plot_drop_log()

    if plot_epochs:
        epochs.plot(scalings=dict(eeg='1e-4', emg='1e-4'))

    return epochs

def scaleEEGPower(powerArray):
    powerArray = powerArray * 1e6**2 
    powerArray = (10 * np.log10(powerArray))
    return powerArray

def epochs_to_dataframe(epochs, label, freq_max=30):
    epochs_power = epochs.compute_psd(fmax=freq_max)
    power_data = epochs_power.get_data()
    power_data = scaleEEGPower(power_data)
    ch_names = BoardShim.get_eeg_names(BOARD)

    # Find an operation that speeds the follown N^3 operation UP 
    df_formatted = pd.DataFrame( {'Channel': [], 'Frequency': [], 'Value': [], 'Label': []})
    for epoch_id, epoch in enumerate(power_data):
            for channel_id, channel_val in enumerate(epoch):
                for freq_index, power in enumerate(channel_val):
                    #print(f'Epoch {epoch_id} channel: {channel_id} freq_index: {freq_index} power: {power}')
                    new_row = {'Channel': ch_names[channel_id], 'Frequency': freq_index, 'Value': power, 'Label': label}
                    df_formatted.loc[len(df_formatted)] = new_row  # len(df) gives the next index number

    return df_formatted

def csv_to_epoch_dataframe(file, event_label, event_id, start=1, stop=5, duration=1.0, freq_max=30, show_muscle_artifacts=False, show_plot=False, show_log=False, drop_bad=True, plot_epochs=False):
    df = csv_to_dataframe(file)
    data = preprocess_data(df, show_muscle_artifacts=show_muscle_artifacts, show_plot=show_plot)
    epochs = get_epochs(data, event_label, event_id, start=start, stop=stop, duration=duration, show_log=show_log, drop_bad=drop_bad, plot_epochs=plot_epochs)
    df = epochs_to_dataframe(epochs, event_label, freq_max=freq_max)
    return df

def main():
    args = get_arguments()
    df = csv_to_epoch_dataframe(args['file_1'], "right_foot", 1, show_muscle_artifacts=True, show_log=False, drop_bad=False, show_plot=False)
    df_2 = csv_to_epoch_dataframe(args['file_2'], "left_foot", 2, show_muscle_artifacts=True)
    combined_df = pd.concat([df, df_2], ignore_index=True)
    df_C3 = combined_df[combined_df['Channel'].isin(['C3'])]
    df_C4 = combined_df[combined_df['Channel'].isin(['C4'])]
    
    plt.clf()
    sns.lineplot(data=df_C3, x="Frequency", y="Value", hue="Label").set(title='C3')
    plt.savefig("c3_foot_serc.png")
    #plt.show()

    plt.clf()
    sns.lineplot(data=df_C4, x="Frequency", y="Value", hue="Label").set(title='C4')
    plt.savefig("c4_foot_serc.png")
    #k=input("press close to exit") 

def csv_to_epochs(file, event_label, event_id):
    df = csv_to_dataframe(file)
    data = preprocess_data(df, show_muscle_artifacts=False, show_plot=False)
    epochs_right_foot = get_epochs(data, event_label, event_id, start=1, stop=5, duration=1.0, show_log=False, drop_bad=True, plot_epochs=False)
    return epochs_right_foot

def combine():
    right_foot_epochs = csv_to_epochs("right_foot_serc.csv", "right_foot", 1)
    print(right_foot_epochs)
    left_foot_epochs = csv_to_epochs("left_foot_serc.csv", "left_foot", 2)
    print(left_foot_epochs)
    combinded_epochs = mne.concatenate_epochs([right_foot_epochs, left_foot_epochs])   
    combinded_epochs.plot(scalings=dict(eeg='1e-4', emg='1e-4')) 
    print(combinded_epochs)
    k=input("press close to exit") 

def run_ica(epochs):
    pass
    # https://mne.tools/stable/generated/mne.preprocessing.create_eog_epochs.html

main()
#combine()

'''
df = csv_to_dataframe(args['file_1'])
data = preprocess_data(df, show_muscle_artifacts=False, show_plot=False)
epochs = get_epochs(data, start=args['epochs_start'], stop=args['epochs_stop'], duration=args['epochs_duration'], show_log=False, drop_bad=True, plot_epochs=False)
df = epochs_to_dataframe(epochs, "right_hand", freq_max=30)
'''
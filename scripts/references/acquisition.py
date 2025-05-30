import argparse
from brainflow import BoardIds
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import annotate_muscle_zscore
import seaborn as sns



data_len = 30
z_score_threshold = 0.0005
freq_max = 20.0

# Note 995 device OSC is not working

def get_flat_columns_channels_freqs(freq_size, epoch_size, channels_array):
    channels = np.array([[[channel for i in range(freq_size)] for channel in channels_array] for x in range(epoch_size)]).reshape(-1)
    freq_bins = np.array([[[i for i in range(freq_size)] for channel in channels_array] for x in range(epoch_size)]).reshape(-1)
    return channels, freq_bins

def scaleEEGPower(powerArray):
    powerArray = powerArray * 1e6**2 
    powerArray = (10 * np.log10(powerArray))
    return powerArray

def dropBadEpochs(epochs, plotLog=False):
    reject_criteria = dict(eeg=300e-6) # 150 µV
    flat_criteria = dict(eeg=1e-6) # 1 µV
    epochs.drop_bad()
    #epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)
    if plotLog: 
        fig = epochs.plot_drop_log()
        fig.savefig('drops_log_epochs.png')

def markMuscleArtifacts(raw, threshold, plotLog=False):
    threshold_muscle = threshold  # z-score
    annot_muscle, scores_muscle = annotate_muscle_zscore(
    raw, ch_type="eeg", threshold=threshold_muscle, min_length_good=0.2, filter_freq=[0, 60])
    raw.set_annotations(annot_muscle)

    if plotLog:
        fig, ax = plt.subplots()
        ax.plot(raw.times, scores_muscle)
        ax.axhline(y=threshold_muscle, color='r')
        ax.set(xlabel='time, (s)', ylabel='zscore', title='Muscle activity')
        fig.savefig('muscle_annot.png')
    


def csvToRaw(file):
    eeg_channels_names = BoardShim.get_eeg_names(BoardIds.CROWN_BOARD.value)
    data = pd.read_csv(file, usecols = eeg_channels_names).transpose()
    sfreq = 256
    data_length = 29

    # import channel info
    ch_names = ['CP3', 'C3', 'F5',  'PO3', 'PO4', 'F6', 'C4',  'CP4' ] 
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', "eeg", "eeg", "eeg", "eeg"]
    info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types=ch_types )
    raw = mne.io.RawArray(data, info)
    markMuscleArtifacts(raw, z_score_threshold)

    # Mark bad dat
    #markMuscleArtifacts(raw, 0.05)

    # High pass at 1hz
    raw_highpass = raw.copy().filter(l_freq=1.0, h_freq=40) 

    # Convert from uV to V for MNE
    raw_highpass.apply_function(lambda x: x * 1e-6)
    #raw.apply_function(lambda x: x * 1e-6)

    # Format data date for annotations later
    raw_highpass.set_meas_date(0)
    raw_highpass.set_montage("standard_1020")

    # Re-reference the data to average
    eeg_reref_highpass, _ = mne.set_eeg_reference(raw_highpass) 
    #fig = eeg_reref_highpass.plot(start=0, duration=20.0, clipping=None)
    #fig.savefig('raw-11.png')

    # Create epochs
    events = [[i * 256, 0, 1] for i in range(data_length)]
    epochs = mne.Epochs(eeg_reref_highpass, events, tmin=0, tmax=1, event_id={"event": 1}, baseline=(0, 0))
    dropBadEpochs(epochs, False)
    #fig = epochs.plot()
    #fig.savefig('epochs-2.png')
    # Calculate Power
    
    epochs_power = epochs.compute_psd(fmax=freq_max)
    power_data = epochs_power.get_data()
    power_data = scaleEEGPower(power_data)
   
    # Find an operation that speeds the follown N^3 operation UP 
    df_formatted = pd.DataFrame( {'Channel': [], 'Frequency': [], 'Value': []})
    for epoch_id, epoch in enumerate(power_data):
            for channel_id, channel_val in enumerate(epoch):
                for freq_index, power in enumerate(channel_val):
                    #print(f'Epoch {epoch_id} channel: {channel_id} freq_index: {freq_index} power: {power}')
                    new_row = {'Channel': ch_names[channel_id], 'Frequency': freq_index, 'Value': power}
                    df_formatted.loc[len(df_formatted)] = new_row  # len(df) gives the next index number
    df_formatted = df_formatted[df_formatted['Channel'].isin(["C4","C3"])]
    sns.lineplot(data=df_formatted, x="Frequency", y="Value", hue="Channel")
    plt.show()
    k=input("press close to exit") 
    

    #print(df_formatted)
    #df_formatted.to_csv('formated_power.csv', index=False)




    #df.to_csv('formated_power_flat.csv', index=False)


    #df_formatted_flat  = pd.DataFrame( {'Channel': [], 'Frequency': [], 'Value': []})
    

    #sns.lineplot(data=df_formatted, x="Frequency", y="Value", hue="Channel")
    #plt.show()
    #k=input("press close to exit") 
    


    #power_2d  = power_data.reshape(-1, power_data.shape[1])    
    #print(power_2d.shape)
    
    
    #power_2d = power_data.reshape(1189, 8)
    #print(power_2d.shape)
    #print(power_2d)
    #df = pd.DataFrame(power_2d)

    #print(df)
    #fig = epochs_power.plot(average=True)
    #fig.savefig('power-1.png')
    

    #return eeg_reref_highpass, raw


def capture(board, file_name="output.csv"):
    eeg_channels_names = BoardShim.get_eeg_names(BoardIds.CROWN_BOARD.value)
    time.sleep(data_len)
    data = board.get_board_data()
    df = pd.DataFrame(np.transpose(data))
    selected = df.loc[:, 1:8]
    selected.columns = eeg_channels_names
    selected.to_csv('crown2.csv', index=False)   

def main():
    params = BrainFlowInputParams()
    #parser = argparse.ArgumentParser()
    #parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    #args = parser.parse_args()
    #params.serial_number = args.serial_number
    #print(args.serial_number)
    #params.serial_number = "ea5a8588f335cd1ceee079f903da4edb"
    
    #BoardShim.enable_dev_board_logger()
    #print(params)
    board = BoardShim(BoardIds.CROWN_BOARD, params)
    board.prepare_session()
    board.start_stream()
    capture(board)

    '''
    while True:
        data = board.get_board_data()
        print(data)
        time.sleep(2)

    #eeg_channels = BoardShim.get_eeg_names(BoardIds.CROWN_BOARD.value)
    #df = pd.DataFrame(np.transpose(data))

    #board.stop_stream()
    #board.release_session()
    #print (data)
    '''



#main()
csvToRaw('crown2.csv')
'''
eeg_channels_names = BoardShim.get_eeg_names(BoardIds.CROWN_BOARD.value)
freq_size = 21
epochs_size = 16
channel_column, freq_bin = get_flat_columns_channels_freqs(freq_size, epochs_size, eeg_channels_names)
'''

#np.savetxt('channel_column.csv', channel_column, delimiter=',', fmt='%s')
#np.savetxt('freq_bins.csv', freq_bin, delimiter=',', fmt='%s')



# n^3 method of formmatting PSD information
'''
df_formatted = pd.DataFrame( {'Channel': [], 'Frequency': [], 'Value': []})
for epoch_id, epoch in enumerate(power_data):
        for channel_id, channel_val in enumerate(epoch):
            for freq_index, power in enumerate(channel_val):
            #print(f'Epoch {epoch_id} channel: {channel_id} freq_index: {freq_index} power: {power}')
            new_row = {'Channel': ch_names[channel_id], 'Frequency': freq_index, 'Value': power}
            df_formatted.loc[len(df_formatted)] = new_row  # len(df) gives the next index number
#df_formatted = df_formatted[df_formatted['Channel'].isin(["C4","C3"])]
#print(df_formatted)
df_formatted.to_csv('formated_power.csv', index=False)
'''
    
'''
    freq_size = power_data.shape[-1]
    epochs_size = power_data.shape[0]
    power_column  = power_data.reshape(-1)
        
    channel_column, freq_bin = get_flat_columns_channels_freqs(freq_size, epochs_size, eeg_channels_names)
    concatenated_array = np.column_stack((channel_column, freq_bin, power_column))
    df = pd.DataFrame(concatenated_array, columns=['Channel', 'Freqs', 'Values'])
    _df = df[df['Channel'].isin(["C4","C3"])]
    sns.lineplot(data=_df, x="Freqs", y="Values", hue="Channel")
    plt.show()
    k=input("press close to exit") 
    
'''


    #np.savetxt('channel_column.csv', channel_column, delimiter=',', fmt='%s')
    #np.savetxt('freq_bins.csv', freq_bin, delimiter=',', fmt='%s')
    #np.savetxt('epochs_2d.csv', power_column, delimiter=',', fmt='%f')

    #start_time = time.time()
    # import channel info
    #ch_names = ['CP3', 'C3', 'F5',  'PO3', 'PO4', 'F6', 'C4',  'CP4' ] 
    
    '''
    ch_types = ['eeg', 'eeg', 'eeg', 'eeg', "eeg", "eeg", "eeg", "eeg"]
    info = mne.create_info(ch_names = eeg_channels_names, sfreq = sfreq, ch_types=ch_types )
    raw = mne.io.RawArray(selected.transpose(), info)
        # High pass at 1hz
    raw_highpass = raw.copy().filter(l_freq=1.0, h_freq=40.0) 

    # Convert from uV to V for MNE
    #raw_highpass.apply_function(lambda x: x * 1e-6)
    raw_highpass.apply_function(lambda x: x * 1e-6)

    # Format data date for annotations later
    raw_highpass.set_meas_date(0)
    raw_highpass.set_montage("standard_1020")

    # Re-reference the data to average
    eeg_reref, _ = mne.set_eeg_reference(raw_highpass) 
    fig = eeg_reref.plot(clipping=None)
    fig.savefig('raw-9.png')
    '''
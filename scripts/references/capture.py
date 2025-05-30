import argparse
from brainflow import BoardIds
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations
import time
import pandas as pd
import numpy as np

def get_arguments():
    argument_info = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('-file_name', type=str, help='Name of file', required=True, default='')
    parser.add_argument('-session_length', type=str, help='Length of recording session', required=False, default='10')

    args = parser.parse_args()
    argument_info['file_name'] = args.file_name
    argument_info['session_length'] = int(args.session_length)

    return argument_info

def get_board():
    params = BrainFlowInputParams()
    board = BoardShim(BoardIds.CROWN_BOARD, params)
    board.prepare_session()
    board.start_stream()
    return board

def capture(board, file_name, data_len=10):
    eeg_channels_names = BoardShim.get_eeg_names(BoardIds.CROWN_BOARD.value)
    print(f'collecing EEG data for {data_len} seconds.')
    time.sleep(data_len)
    data = board.get_board_data()
    print("Data collection complete.")
    df = pd.DataFrame(np.transpose(data))
    selected = df.loc[:, 1:8]
    selected.columns = eeg_channels_names
    selected.to_csv(file_name, index=False)   

def main():
    args = get_arguments()
    board = get_board()
    capture(board, args['file_name'], args['session_length'])

main()    

'''
from mne.datasets import eegbci
tmin, tmax = -1.0, 4.0
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet
from mne.io import concatenate_raws, read_raw_edf

raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
#raw.plot()
print(raw.info)
k=input("press close to exit") 
'''

#print(raw)


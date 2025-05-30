from htil_eeg import HTIL_EEG 
from brainflow import BoardIds
import numpy as np
import mne
from utils import events, sound, files
import time

duration = 1000  # milliseconds
freq = 440  # Hz
#h_eeg = HTIL_EEG(BoardIds.CROWN_BOARD)

def run_experiment_continuous(experiment_name, total_epochs=2, epoch_length=2):
    _events = []
    k=input("Start experiment")
    #event_dict = {"closed": 1}
    event_labels = events.create_event_array([ "closed", "left", "right", "open"], [0.25, 0.25, 0.25, 0.25], total_epochs, shuffle=True)
    #h_eeg.start_stream()
    start_time = time.time()
    for event in event_labels:
        print(f"______________")
        print(f"stimulus {event}")
        #if event == "open":
            #sound.play_sound(freq, duration)
        time.sleep(epoch_length)
        #_events.append(events.create_event(start_time, event_dict[event], 256))
    sound.play_sound(freq, duration)
    #df, processed, raw = h_eeg.get_recent_crown_data()
    #np_events = np.array(_events)
    #epochs = mne.Epochs(processed, np_events, tmin=-1.0, tmax=1.0, event_id=event_dict, baseline=(-0.5, 0))
    #files.save(df, raw, epochs, np_events, experiment_name)


def run_experiment(experiment_name, total_epochs=2, epoch_length=2):
    _events = []
    k=input("Start experiment")
    event_dict = {"left": 1, "right": 2, "rest": 3}
    event_labels = events.create_event_array(["left", "right"], [0.5, 0.5], total_epochs, shuffle=True)
    h_eeg.start_stream()
    start_time = time.time()
    for event in event_labels:
        print(f"______________")
        print(f"stimulus {event}")
        #if event == "open":
            #sound.play_sound(freq, duration)
        time.sleep(epoch_length)
        _events.append(events.create_event(start_time, event_dict[event], 256))
        
        print(f"stimulus rest")
        time.sleep(epoch_length)
        _events.append(events.create_event(start_time, event_dict["rest"], 256))



    df, processed, raw = h_eeg.get_recent_crown_data()
    np_events = np.array(_events)
    epochs = mne.Epochs(processed, np_events, tmin=-1.0, tmax=1.0, event_id=event_dict, baseline=(-0.5, 0))
    files.save(df, raw, epochs, np_events, experiment_name)

run_experiment_continuous("closed_7", total_epochs=20, epoch_length=3)
#run_experiment_eyes("open_1", total_epochs=10, epoch_length=2)
#run_experiment("left_right_2", total_epochs=30, epoch_length=2)
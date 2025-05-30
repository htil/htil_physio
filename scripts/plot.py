from sklearn.pipeline import Pipeline
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from mne import Epochs, pick_types
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.decoding import CSP
from mne.io import concatenate_raws, read_raw_edf

from htil_eeg import HTIL_EEG 
from brainflow import BoardIds

h_eeg = HTIL_EEG(BoardIds.CROWN_BOARD, real_time=False, z_score_threshold=0.25)
raw_file_path = "open_1"
events_file_path = "open_1"

event_dict = {"open": 1}
events_colors =  {1:"green"}
l_freq=2
h_freq=30
show_muscle_artifacts=False
epochs = h_eeg.raw_events_to_epochs(f"../data/raw/{raw_file_path}_raw.fif", f"../data/events/{events_file_path}_events.csv", event_dict, events_colors, l_freq=l_freq, h_freq=h_freq, show_muscle_artifacts=show_muscle_artifacts)

plt.clf()
events_colors =  {1:"green", 2: "red"}
epochs.plot(event_id=True, events=True, event_color=events_colors)
plt.show()
from htil_eeg import HTIL_EEG 
import time
from brainflow import BoardIds


def test_connection():
    print("Testing connection...")
    h_eeg = HTIL_EEG(BoardIds.CROWN_BOARD)
    h_eeg.start_stream()
    time.sleep(20)
    df, processed, raw = h_eeg.get_recent_crown_data()
    print(df)
    # h_eeg.stop_stream()
    print("Connection test complete")

test_connection()

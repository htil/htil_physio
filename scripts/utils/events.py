
import random
import math
import time

def create_event_array(event_labels, event_frequency, total_epochs, shuffle=True):
    events = []
    for index, label in enumerate(event_labels):
        freq = round(total_epochs * event_frequency[index])
        for j in range(freq):
            events.append(label)
    if shuffle:
        random.shuffle(events)
    return events

def create_event(start_time, label, sfreq):
    elapsed = math.floor((time.time() - start_time) * sfreq)
    return [elapsed, 0, label]
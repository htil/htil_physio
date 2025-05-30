import numpy as np
from pathlib import Path 

def save(df, raw, epochs, np_events, experiment_name):
    df.to_csv(f'../data/csv/{experiment_name}.csv', index=False)   
    raw.save(Path(f'../data/raw/{experiment_name}_raw.fif'), overwrite=True)
    epochs.save(Path('../data/epochs') / f'{experiment_name}_epo.fif', overwrite=True)
    np.savetxt(f"../data/events/{experiment_name}_events.csv", np_events, delimiter=',', fmt='%d')
    k=input("Press enter to continue.")
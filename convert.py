from obspy import read
stream = read("your_file.mseed")

import h5py
import numpy as np

# Open a new HDF5 file
with h5py.File('output.h5', 'w') as h5f:
    for i, trace in enumerate(stream):
        grp = h5f.create_group(f"trace_{i}")
        grp.create_dataset('data', data=trace.data)
        grp.attrs['starttime'] = str(trace.stats.starttime)
        grp.attrs['endtime'] = str(trace.stats.endtime)
        grp.attrs['sampling_rate'] = trace.stats.sampling_rate

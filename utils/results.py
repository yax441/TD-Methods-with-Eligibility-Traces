import pickle
import numpy as np

def save_results(path, file_name, data):
    with open(f'{path}/{file_name}.ob', 'wb') as f:
        pickle.dump(data, f)
        
def load_results(path, file_name):
    with open (f'{path}/{file_name}.ob', 'rb') as f:
        return pickle.load(f)

def flt(data, operation='mean', last=None, first=None, skip_first=True, dim1_filter=None):
    fun = None
    if operation == 'mean':
        fun = np.mean
    elif operation == 'max':
        fun = np.max
    elif operation == 'min':
        fun = np.min
    data = np.array(data)[dim1_filter] if dim1_filter is not None else np.array(data)
    shift = 1 if skip_first else 0
    if first is not None and last is not None:
        return fun(data[:,:,first:last+1], axis=2) if fun else data[:,:,first:last+1]
    if last is not None:
        return fun(data[:,:,-last:], axis=2) if fun else data[:,:,-last:]
    elif first is not None:
        return fun(data[:,:,shift:first+1+shift], axis=2) if fun else data[:,:,shift:first+1+shift]
    return fun(data[:,:,shift:], axis=2) if fun else data[:,:,shift:]
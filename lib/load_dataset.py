import os
import numpy as np
import pandas as pd
import pickle

def load_st_dataset(args):
    dataset = args.dataset
    if 'PeMSD4FLOW' in dataset:
        data = np.load('./data/PeMSD4/pems04.npz')
        data = data['data']
        data = data[:,:,0]
    elif 'PeMSD4OCCUPANCY' in dataset:
        data = np.load('./data/PeMSD4/pems04.npz')
        data = data['data']
        data = data[:,:,1]
    elif 'PeMSD4SPEED' in dataset:
        data = np.load('./data/PeMSD4/pems04.npz')
        data = data['data']
        data = data[:,:,2]
    elif 'PeMSD7' in dataset:
        df = pd.read_csv('./data/PeMSD7/data.csv')
        data = df.drop(columns='time').to_numpy(dtype=np.float64)
    elif 'PeMSD8FLOW' in dataset:
        data = np.load('./data/PeMSD8/pems08.npz')
        data = data['data']
        data = data[:,:,0]
    elif 'PeMSD8OCCUPANCY' in dataset:
        data = np.load('./data/PeMSD8/pems08.npz')
        data = data['data']
        data = data[:,:,1]
    elif 'PeMSD8SPEED' in dataset:
        data = np.load('./data/PeMSD8/pems08.npz')
        data = data['data']
        data = data[:,:,2]
    elif 'METR_LA' in dataset:
        df = pd.read_hdf(f"./data/METR_LA/metr-la.h5")
        data = df.to_numpy()
    elif 'PEMS_BAY' in dataset:
        df = pd.read_hdf(f"./data/PEMS_BAY/pems-bay.h5")
        data = df.to_numpy()
    else:
        raise ValueError
    data = data[:, args.nodes]
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

import os
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_nsmc_set():
    import pandas as pd
    train = pd.read_csv("data/ratings_train.txt", sep='\t')
    test = pd.read_csv("data/ratings_test.txt", sep='\t')

    return train, test

def convert_to_tensor(inputs, labels, masks, batch_size = 32):
    tensor_inputs = torch.tensor(inputs)
    tensor_labels = torch.tensor(labels)
    tensor_masks = torch.tensor(masks)

    twined_data = TensorDataset(tensor_inputs, tensor_masks, tensor_labels)
    data_sampler = RandomSampler(twined_data)
    tensor_dataloader = DataLoader(twined_data, sampler=data_sampler, batch_size=batch_size)

    return tensor_dataloader


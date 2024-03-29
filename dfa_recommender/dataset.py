import torch
import torch.utils.data


class SubsetDataset(torch.utils.data.Dataset):
    '''
    Subset a torch.utils.data.Dataset object
    '''
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

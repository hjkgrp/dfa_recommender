import numpy as np
from torch.utils import data


class InfiniteSampler(data.sampler.Sampler):
    '''
    Sample datasets
    '''
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        np.random.seed(0)
        # i = self.num_samples - 1
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                # np.random.seed(0)
                order = np.random.permutation(self.num_samples)
                i = 0

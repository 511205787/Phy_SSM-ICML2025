import numpy as np
from torch.utils.data import Dataset
import torch
import random
import matplotlib.pyplot as plt

class ODEDataSet_with_control(Dataset):
    def __init__(self, file_path, ds_type, seq_len, random_start, discard_fraction=0.2, transforms=None):
        self.transforms = transforms if transforms is not None else {}
        self.random_start = random_start
        self.ds_type = ds_type
        self.seq_len = seq_len
        self.discard_fraction = discard_fraction

        data_dict = torch.load(file_path)

        if ds_type == 'train':
            buffer = int(round(data_dict["train"].shape[0] * (1 - 0.1)))
            self.noisy_data, self.control_data, self.latent_data, self.clean_data, self.t = self._process_data(
                torch.FloatTensor(data_dict["train"][:buffer]),
                torch.FloatTensor(data_dict["train_control"][:buffer]),
                torch.FloatTensor(data_dict["train_latent"][:buffer]),
                torch.FloatTensor(data_dict["raw_train"][:buffer])
            )
        elif ds_type == 'val':
            buffer = int(round(data_dict["train"].shape[0] * (1 - 0.1)))
            self.noisy_data, self.control_data, self.latent_data, self.clean_data, self.t = self._process_data(
                torch.FloatTensor(data_dict["train"][buffer:]),
                torch.FloatTensor(data_dict["train_control"][buffer:]),
                torch.FloatTensor(data_dict["train_latent"][buffer:]),
                torch.FloatTensor(data_dict["raw_train"][buffer:])
            )
        elif ds_type == 'test':
            self.noisy_data, self.control_data, self.latent_data, self.clean_data, self.t = self._process_data(
                torch.FloatTensor(data_dict["test"]),
                torch.FloatTensor(data_dict["test_control"]),
                torch.FloatTensor(data_dict["test_latent"]),
                torch.FloatTensor(data_dict["raw_test"])
            )

    def __len__(self):
        return self.noisy_data.size(0)

    def __getitem__(self, idx):

        if self.random_start:
            start_time = random.randint(0, self.noisy_data.size(1) - self.seq_len)
            noisy_sample = self.noisy_data[idx, start_time:start_time + self.seq_len]
            control_sample = self.control_data[idx, start_time:start_time + self.seq_len]
            latent_sample = self.latent_data[idx, start_time:start_time + self.seq_len]
            clean_sample = self.clean_data[idx, start_time:start_time + self.seq_len]
            t_sample = self.t[idx, start_time:start_time + self.seq_len]
        else:
            noisy_sample = self.noisy_data[idx]
            control_sample = self.control_data[idx]
            latent_sample = self.latent_data[idx]
            clean_sample = self.clean_data[idx]
            t_sample = self.t[idx]

        # Plot the control sample
        # plt.figure()
        # if control_sample.dim() == 2:
        #     for i in range(control_sample.size(1)):  # Assuming each column is a separate feature
        #         plt.plot(t_sample, control_sample[:, i], label=f'Control Feature {i}')
        # else:
        #     plt.plot(t_sample, control_sample, label='Control Data')
        
        # plt.title(f'Control Data Sample {idx}')
        # plt.xlabel('Time')
        # plt.ylabel('Control Value')
        # plt.legend()
        # plt.savefig(f'control_sample_{idx}.png')
        # plt.close()

        for transform in self.transforms:
            noisy_sample = self.transforms[transform](noisy_sample)
            clean_sample = self.transforms[transform](clean_sample)

        return noisy_sample, control_sample, latent_sample, clean_sample, t_sample

    def _process_data(self, noisy_data, control_data, latent_data, clean_data):
        processed_noisy_data = []
        processed_control_data = []
        processed_latent_data = []
        processed_clean_data = []
        t_data = []

        for i in range(noisy_data.size(0)):
            original_seq_len = noisy_data.size(1)
            discard_length = int(original_seq_len * self.discard_fraction)
            discard_indices = sorted(random.sample(range(original_seq_len), discard_length))
            keep_indices = [i for i in range(original_seq_len) if i not in discard_indices]

            t = torch.arange(0, original_seq_len * 0.05, 0.05)
            t_keep = t[keep_indices]

            processed_noisy_data.append(noisy_data[i, keep_indices, ...])
            processed_control_data.append(control_data[i, keep_indices, ...])
            processed_latent_data.append(latent_data[i, keep_indices, ...])
            processed_clean_data.append(clean_data[i, keep_indices, ...])
            t_data.append(t_keep)

        return (torch.stack(processed_noisy_data), torch.stack(processed_control_data),
                torch.stack(processed_latent_data), torch.stack(processed_clean_data),
                torch.stack(t_data))

class NormalizeZScore(object):
    """Normalize sample by mean and std."""
    def __init__(self, data_norm_params):
        self.mean = torch.FloatTensor(data_norm_params["mean"])
        self.std = torch.FloatTensor(data_norm_params["std"])

    def __call__(self, sample):
        new_sample = torch.zeros_like(sample, dtype=torch.float)
        for feature in range(self.mean.size(0)):
            if self.std[feature] > 0:
                new_sample[:, feature] = (sample[:, feature] - self.mean[feature]) / self.std[feature]
            else:
                new_sample[:, feature] = (sample[:, feature] - self.mean[feature])

        return new_sample

    def denormalize(self, batch):
        denormed_batch = torch.zeros_like(batch)
        for feature in range(batch.size(2)):
            denormed_batch[:, :, feature] = (batch[:, :, feature] * self.std[feature]) + self.mean[feature]

        return denormed_batch


class NormalizeToUnitSegment(object):
    """Normalize sample to the segment [0, 1] by max and min"""
    def __init__(self, data_norm_params):
        self.min_val = data_norm_params["min"]
        self.max_val = data_norm_params["max"]

    def __call__(self, sample):
        new_sample = (sample - self.min_val) / (self.max_val - self.min_val)
        return new_sample

    def denormalize(self, batch):
        denormed_batch = (batch * (self.max_val - self.min_val)) + self.min_val
        return denormed_batch


import numpy as np
from torch.utils.data import Dataset
import torch
import random

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class SIRDataSet(Dataset):
    def __init__(self, data_dir, countries, mean=None, std=None, training_mode=True, calculate_stats=False):
        self.calculate_stats = calculate_stats
        self.discard_fraction = 0.1
        self.training_mode = training_mode
        # Load all country data and find the minimum sequence length
        self.data_list = []
        min_seq_len = float('inf')
        for country in countries:
            file_path = os.path.join(data_dir, f"sir_data_{country}.csv")
            data = pd.read_csv(file_path)

            # Drop the first column (assuming it's the date)
            data = data.drop(columns=data.columns[0])

            # Remove non-numeric columns
            data = data.select_dtypes(include=[np.number])

            # Handle any NaN or None values by filling them with 0 (or you can use data.fillna(data.mean()) for mean)
            data = data.fillna(data.mean())

            # Convert to tensor
            data_tensor = torch.tensor(data.values, dtype=torch.float32)

            norm_factor = data_tensor[0, 0] + data_tensor[0, 1] + data_tensor[0, 2]
            data_tensor = data_tensor / norm_factor

            self.data_list.append(data_tensor)

            if data_tensor.size(0) < min_seq_len:
                min_seq_len = data_tensor.size(0)

        # self.min_seq_len = min_seq_len
        self.min_seq_len = 300

        # Truncate all data to the minimum sequence length
        self.truncated_data = [data[:self.min_seq_len] for data in self.data_list]

        # Stack all data into a single tensor with shape [num_countries, min_seq_len, 3]
        self.all_data = torch.stack(self.truncated_data)

        # Calculate the mean and std for the combined data if required
        if calculate_stats:
            self.mean = self.all_data.mean(dim=(0, 1), keepdim=True)
            self.std = self.all_data.std(dim=(0, 1), keepdim=True)
        else:
            self.mean = mean
            self.std = std

        # Apply the standardization
        self.all_data = (self.all_data - self.mean) / (self.std + 1e-8)

        # Process data to generate input-output pairs with slicing
        self.all_data, self.t = self._process_data(self.all_data)

    def __len__(self):
        # Return the number of sequences (after slicing) across all countries
        return self.all_data.size(0)

    def __getitem__(self, idx):
        # Return the input sequence, output sequence, and corresponding time steps
        input_sequence = self.all_data[idx][:-1]  # First 200 elements (0 to 199)
        output_sequence = self.all_data[idx][1:]  # Next 200 elements (1 to 200)
        t_sample = self.t[idx][:-1]
        return input_sequence, output_sequence, output_sequence, output_sequence, t_sample
    
    def _process_data(self, noisy_data):
        processed_noisy_data = []
        t_data = []
        seg_len = 240
        if self.training_mode:  # Only slice sequences when in training mode
            for i in range(noisy_data.size(0)):
                original_seq_len = noisy_data.size(1)
                discard_length = int(original_seq_len * self.discard_fraction)
                discard_indices = sorted(random.sample(range(original_seq_len), discard_length))
                keep_indices = [i for i in range(original_seq_len) if i not in discard_indices]

                t = torch.arange(0, original_seq_len * 0.05, 0.05)
                t_keep = t[keep_indices]
                noisy_data_keep = noisy_data[i, keep_indices, ...]

                # Slice the sequences into input-output pairs with length 200
                for j in range(len(keep_indices) - (seg_len)):
                    input_seq = noisy_data_keep[j:j + (seg_len + 1)]  # Slice length 201
                    processed_noisy_data.append(input_seq)
                    t_seq = t_keep[j:j + (seg_len + 1)]  # Corresponding time sequence
                    t_data.append(t_seq)
        else:
            for i in range(noisy_data.size(0)):
                original_seq_len = noisy_data.size(1)
                discard_length = int(original_seq_len * self.discard_fraction)
                discard_indices = sorted(random.sample(range(original_seq_len), discard_length))
                keep_indices = [i for i in range(original_seq_len) if i not in discard_indices]

                t = torch.arange(0, original_seq_len * 0.05, 0.05)
                t_keep = t[keep_indices]
                noisy_data_keep = noisy_data[i, keep_indices, ...]
                noisy_data_keep = noisy_data_keep[:seg_len]
                t_keep = t_keep[:seg_len]
                processed_noisy_data.append(noisy_data_keep)
                t_data.append(t_keep)
        # Stack all the processed sequences into a new tensor
        return torch.stack(processed_noisy_data), torch.stack(t_data)

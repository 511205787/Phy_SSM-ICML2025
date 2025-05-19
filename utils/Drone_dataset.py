import os
import pickle
import numpy as np
from torch.utils.data import Dataset

class DroneFlightDataset(Dataset):
    def __init__(self, data_dir, ds_type='train', seq_len=100, discard_fraction=0, transform=None, stats=None, feature_indices=None):
        """
        Args:
            data_dir (string): Directory with all the pickle files.
            ds_type (string): One of 'train', 'valid', 'test' to specify the dataset split.
            seq_len (int): Length of each sequence after discarding.
            discard_fraction (float): Fraction of data to discard (fixed for all sequences). Indices to discard are randomly selected.
            transform (callable, optional): Optional transform to be applied on a sample.
            stats (dict, optional): Dictionary containing 'mean' and 'std' for normalization. If None, compute from training data.
            feature_indices (list, optional): List of tuples indicating feature indices for normalization. If None, compute from data.
        """
        self.data_dir = data_dir
        self.ds_type = ds_type
        self.seq_len = seq_len
        self.discard_fraction = discard_fraction
        self.transform = transform
        self.stats = stats  # Statistics for normalization
        self.feature_indices = feature_indices  # Feature indices for normalization
        self.sequences = []  # List to store sequences
        self.load_data()
        self.prepare_sequences()
        self.split_dataset()
        self.compute_feature_indices()
        if self.ds_type == 'train' and self.stats is None:
            self.compute_stats()
        elif self.stats is not None:
            # Use provided stats
            self.mean = self.stats['mean']
            self.std = self.stats['std']
        else:
            raise ValueError("Stats must be provided for validation and test datasets")
        # Normalize data during initialization
        self.normalize_sequences()

    def load_data(self):
        # Load all pickle files
        self.all_data = []  # List to store data from all pickle files
        pickle_files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.pkl')])
        for pickle_file in pickle_files:
            file_path = os.path.join(self.data_dir, pickle_file)
            with open(file_path, 'rb') as f:
                combined_data = pickle.load(f)
                self.all_data.append(combined_data)

    def prepare_sequences(self):
        # For each combined_data in all_data, process the data
        sequences = []
        for data in self.all_data:
            num_samples = len(data['timestamps'])
            if num_samples < self.seq_len:
                continue  # Not enough data
            # Calculate discard length
            discard_length = int(num_samples * self.discard_fraction)
            # Randomly select indices to discard
            if discard_length > 0:
                discard_indices = sorted(np.random.choice(num_samples, discard_length, replace=False))
                # Indices to keep
                keep_indices = np.array([i for i in range(num_samples) if i not in discard_indices])
            else:
                keep_indices = np.arange(num_samples)
            # Sort the kept indices in ascending order to maintain time order
            keep_indices.sort()
            # Get the data for kept indices
            kept_data = {}
            for key in data.keys():
                kept_data[key] = data[key][keep_indices]
            # Now, slice the kept data into sequences of length seq_len
            num_kept_samples = len(keep_indices)
            if num_kept_samples < self.seq_len:
                continue  # Not enough data after discarding
            # Starting indices for sequences (overlapping sequences)
            seq_starts = np.arange(0, num_kept_samples - self.seq_len + 1)
            for start_idx in seq_starts:
                end_idx = start_idx + self.seq_len
                seq_indices = np.arange(start_idx, end_idx)
                sequence = {}
                for key in kept_data.keys():
                    if key == 'timestamps':
                        seq_data = kept_data[key][seq_indices]
                        seq_data = seq_data - seq_data[0]  # Reset timestamps to start from 0
                        sequence[key] = seq_data
                    else:
                        sequence[key] = kept_data[key][seq_indices]
                sequences.append(sequence)
        self.sequences = sequences

    def split_dataset(self):
        # Shuffle the sequences
        np.random.shuffle(self.sequences)
        # Split into train, valid, test
        total_sequences = len(self.sequences)
        train_end = int(0.7 * total_sequences)
        valid_end = train_end + int(0.1 * total_sequences)
        if self.ds_type == 'train':
            self.sequences = self.sequences[:train_end]
        elif self.ds_type == 'valid':
            self.sequences = self.sequences[train_end:valid_end]
        elif self.ds_type == 'test':
            self.sequences = self.sequences[valid_end:]
        else:
            raise ValueError("Invalid ds_type. Should be 'train', 'valid', or 'test'")

    def compute_feature_indices(self):
        # Compute feature indices based on the feature dimensions
        feature_dims = []
        self.feature_keys = []
        for key in self.sequences[0].keys():
            if key != 'timestamps':
                feature_dims.append(self.sequences[0][key].shape[-1])
                self.feature_keys.append(key)
        self.feature_indices = []
        cum_size = 0
        for dim in feature_dims:
            self.feature_indices.append((cum_size, cum_size + dim))
            cum_size += dim
        self.total_feature_dim = cum_size  # Total dimension of all features concatenated

    def compute_stats(self):
        # Compute mean and std of features over the training data incrementally
        print("Computing mean and std for normalization...")
        n_samples = 0
        feature_sums = np.zeros(self.total_feature_dim)
        feature_square_sums = np.zeros(self.total_feature_dim)

        for sequence in self.sequences:
            # Concatenate all features except 'timestamps'
            features = []
            for key in self.feature_keys:
                value = sequence[key]
                features.append(value.reshape(-1, value.shape[-1]))
            # Concatenate along the feature dimension
            sequence_features = np.concatenate(features, axis=1)  # Shape: [seq_len, total_features]
            # Update sums
            feature_sums += np.sum(sequence_features, axis=0)
            feature_square_sums += np.sum(sequence_features ** 2, axis=0)
            n_samples += sequence_features.shape[0]
        # Compute mean and std
        self.mean = feature_sums / n_samples
        variance = (feature_square_sums / n_samples) - (self.mean ** 2)
        self.std = np.sqrt(variance)
        # To avoid division by zero
        self.std[self.std == 0] = 1
        # Store stats
        self.stats = {'mean': self.mean, 'std': self.std}

    def normalize_sequences(self):
        # Normalize features in self.sequences using computed mean and std
        print(f"Normalizing data for {self.ds_type} dataset...")
        for sequence in self.sequences:
            feature_idx = 0
            for key in self.feature_keys:
                value = sequence[key]
                value_shape = value.shape
                value_flat = value.reshape(-1, value_shape[-1])
                idx_start, idx_end = self.feature_indices[feature_idx]
                mean = self.mean[idx_start:idx_end]
                std = self.std[idx_start:idx_end]
                normalized_value = (value_flat - mean) / std
                normalized_value = normalized_value.reshape(value_shape)
                sequence[key] = normalized_value
                feature_idx += 1  # Move to next feature
            # 'timestamps' remain unnormalized
        # After normalization, sequences are ready for use

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample = self.sequences[idx]
        rpms = sample['rpms'].astype(np.float32) # shape N, 4
        acceleration = sample['acceleration'].astype(np.float32) # shape N, 3
        omega = sample['omega'].astype(np.float32) # shape N, 3
        domega = sample['domega'].astype(np.float32) # shape N, 3
        thrust_values = sample['thrust_values'][:, 2].astype(np.float32)
        geometric_torque_values = sample['geometric_torque_values'][:, :-1].astype(np.float32)
        # input_sequence = np.concatenate(
        # (acceleration, omega, domega, rpms, thrust_values[:, None], geometric_torque_values),
        # axis=1).astype(np.float32)
        input_sequence = np.concatenate((acceleration, omega, domega), axis=1).astype(np.float32)
        control_data = np.concatenate((rpms, thrust_values[:, None], geometric_torque_values), axis=1).astype(np.float32)
        timestamps = sample['timestamps'][1:].astype(np.float32)
        output_sequence = input_sequence[1:].astype(np.float32)
        return input_sequence[:-1], control_data[:-1], output_sequence, output_sequence, timestamps

    def get_stats(self):
        return self.stats

    def get_feature_indices(self):
        return self.feature_indices

if __name__ == '__main__':
    # Instantiate the dataset for training
    train_dataset = DroneFlightDataset(
        data_dir='data/drone_data',
        ds_type='train',
        seq_len=1001,
        discard_fraction=0.1  # Adjust discard_fraction as needed
    )
    # Get the computed stats and feature_indices
    stats = train_dataset.get_stats()
    feature_indices = train_dataset.get_feature_indices()

    # Instantiate the dataset for validation and testing, using the same stats and feature_indices
    valid_dataset = DroneFlightDataset(
        data_dir='data/drone_data',
        ds_type='valid',
        seq_len=1001,
        discard_fraction=0.1,
        stats=stats,  # Use the stats from training data
        feature_indices=feature_indices  # Use feature indices from training data
    )
    test_dataset = DroneFlightDataset(
        data_dir='data/drone_data',
        ds_type='test',
        seq_len=1001,
        discard_fraction=0.1,
        stats=stats,  # Use the stats from training data
        feature_indices=feature_indices  # Use feature indices from training data
    )

    # Use with DataLoader for batching
    from torch.utils.data import DataLoader
    import torch

    def collate_fn(batch):
        # Since sequences are of fixed length, we can stack them directly
        batch_timestamps = torch.tensor([sample['timestamps'] for sample in batch], dtype=torch.float32)
        batch_acceleration = torch.tensor([sample['acceleration'] for sample in batch], dtype=torch.float32)
        batch_rpms = torch.tensor([sample['rpms'] for sample in batch], dtype=torch.float32)
        # Add other features as needed
        return {
            'timestamps': batch_timestamps,
            'acceleration': batch_acceleration,
            'rpms': batch_rpms,
            # Add other keys
        }

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Iterate over batches
    for batch in dataloader:
        print(batch)
        break  # Remove this break statement in actual use

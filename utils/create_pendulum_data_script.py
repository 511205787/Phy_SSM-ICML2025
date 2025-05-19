import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import argparse
import torch
from utils.utils import set_seed
import os
from ..config import load_data_config
from utils.pendulum_equ import create_pendulum_data
import matplotlib.pyplot as plt
# Reference: https://github.com/orilinial/GOKU
# taken from https://github.com/ALRhub/rkn_share/ and modified the initial factors range
class NoiseAdder:
    def __init__(self, random_state=None):
        self.random = np.random.RandomState(random_state)

    def add_img_noise(self, imgs, first_n_clean, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0):
        assert t_ll <= t_lu <= t_ul <= t_uu, "Invalid bounds for noise generation"
        if len(imgs.shape) < 5:
            imgs = np.expand_dims(imgs, -1)
        batch_size, seq_len = imgs.shape[:2]
        factors = np.zeros([batch_size, seq_len])
        # factors[:, 0] = self.random.uniform(low=0.0, high=1.0, size=batch_size)
        factors[:, 0] = self.random.uniform(low=0.50, high=1.0, size=batch_size)
        for i in range(seq_len - 1):
            factors[:, i + 1] = np.clip(factors[:, i] + self.random.uniform(
                low=-r, high=r, size=batch_size), a_min=0.0, a_max=1.0)

        t1 = self.random.uniform(low=t_ll, high=t_lu, size=(batch_size, 1))
        t2 = self.random.uniform(low=t_ul, high=t_uu, size=(batch_size, 1))

        factors = (factors - t1) / (t2 - t1)
        # factors = np.clip(factors, a_min=0.0, a_max=1.0)
        factors = np.clip(factors, a_min=0.50, a_max=1.0)
        factors = np.reshape(factors, list(factors.shape) + [1, 1, 1])
        factors[:, :first_n_clean] = 1.0
        noisy_imgs = []

        for i in range(batch_size):
            if imgs.dtype == np.uint8:
                noise = self.random.uniform(low=0.0, high=255, size=imgs.shape[1:])
                noisy_imgs.append(
                    (factors[i] * imgs[i] + (1 - factors[i]) * noise).astype(np.uint8))
            else:
                noise = self.random.uniform(low=0.0, high=1.0, size=imgs.shape[1:])
                noisy_imgs.append(factors[i] * imgs[i] + (1 - factors[i]) * noise)

        return np.squeeze(np.concatenate([np.expand_dims(n, 0) for n in noisy_imgs], 0)), factors

def find_norm_params(data):
    mean = np.zeros(data.shape[2])
    std = np.zeros(data.shape[2])
    for feature in range(data.shape[2]):
        mean[feature] = data[:, :, feature].mean()
        std[feature] = data[:, :, feature].std()

    max_val = data.max()
    min_val = data.min()

    data_norm_params = {"mean": mean,
                        "std" : std,
                        "max" : max_val,
                        "min" : min_val}

    return data_norm_params

def normalize_data(data, params):
    data = (data - params['min']) / (params['max'] - params['min'])
    return data

def denormalize_data(data, params):
    data = data * (params['max'] - params['min']) + params['min']
    return data

def save_noisy_image(args, original_data, noisy_data, batch=5):
    os.makedirs(args.output_dir, exist_ok=True)
    fig, ax = plt.subplots(batch, 2, figsize=(10, 5 * batch))

    for i in range(batch):
        # Display the original image sequence
        original_sequence = original_data[i]
        noisy_sequence = noisy_data[i]

        num_frames = original_sequence.shape[0]
        original_frames = [original_sequence[j, :, :] for j in range(num_frames)]
        noisy_frames = [noisy_sequence[j, :, :] for j in range(num_frames)]

        original_image_sequence = np.hstack(original_frames)
        noisy_image_sequence = np.hstack(noisy_frames)

        ax[i, 0].imshow(original_image_sequence, cmap='gray')
        ax[i, 0].set_title(f'Original Image Sequence {i+1}')

        ax[i, 1].imshow(noisy_image_sequence, cmap='gray')
        ax[i, 1].set_title(f'Noisy Image Sequence {i+1}')

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'noisy_image_sequences.png'))
    plt.close()

def add_Gaussion_noise(args, data, params):
    normalized_data = normalize_data(data, params)
    noisy_data = normalized_data + args.noise_std * np.random.normal(size=normalized_data.shape)
    noisy_data = np.clip(noisy_data, 0, 1) 
    noisy_data = denormalize_data(noisy_data, params)
    save_noisy_image(args, data, noisy_data, batch=5)
    return noisy_data

def add_noise(args, data, params):
    normalized_data = normalize_data(data, params)
    noise_adder = NoiseAdder(random_state=args.seed)
    noisy_data, _ = noise_adder.add_img_noise(normalized_data, first_n_clean=0, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)
    noisy_data = np.clip(noisy_data, 0, 1) 
    noisy_data = denormalize_data(noisy_data, params)
    save_noisy_image(args, data, noisy_data, batch=5)
    return noisy_data

def create_mask(args, data_shape):
    revealed_n = int(round(args.mask_rate * args.seq_len))
    latent_mask = np.zeros(data_shape)
    max_val = int(args.seq_len * 0.75)
    for sample in range(data_shape[0]):
        train_latent_mask_ind = np.random.choice(range(max_val), size=revealed_n, replace=False)
        for mask_idx in train_latent_mask_ind:
            latent_mask[sample, mask_idx, :] = 1

    return latent_mask


def make_dataset(args):
    if not args.change_only_mask_rate:
        raw_data, control_data, latent_data, params_data = args.create_raw_data(args)

        buffer = int(round(raw_data.shape[0] * (1 - 0.1)))

        train_data = raw_data[:buffer]
        test_data = raw_data[buffer:]

        data_norm_params = find_norm_params(train_data)  

        noisy_train_data = add_Gaussion_noise(args, train_data, data_norm_params)
        noisy_test_data = add_Gaussion_noise(args, test_data, data_norm_params)

        train_latent_data = latent_data[:buffer]
        test_latent_data = latent_data[buffer:]

        train_control_data = control_data[:buffer] 
        test_control_data = control_data[buffer:]  

        train_params_data = params_data[:buffer]
        train_params_data = {key: np.array([sample[key] for sample in train_params_data]) for key in train_params_data[0]}

        test_params_data = params_data[buffer:]
        test_params_data = {key: np.array([sample[key] for sample in test_params_data]) for key in test_params_data[0]}

        torch.save(train_params_data, args.output_dir + 'train_params_data.pkl')
        torch.save(test_params_data, args.output_dir + 'test_params_data.pkl')
        torch.save(train_latent_data, args.output_dir + 'train_latent_data.pkl')
        torch.save(test_latent_data, args.output_dir + 'test_latent_data.pkl')
        torch.save(data_norm_params, args.output_dir + 'data_norm_params.pkl')
        torch.save(test_data, args.output_dir + 'gt_test_data.pkl')
        torch.save(train_control_data, args.output_dir + 'train_control_data.pkl')  # 新增：保存控制信号的训练数据
        torch.save(test_control_data, args.output_dir + 'test_control_data.pkl')  # 新增：保存控制信号的测试数据

    else:
        train_latent_data = torch.load(args.output_dir + 'train_latent_data.pkl')
        test_latent_data = torch.load(args.output_dir + 'test_latent_data.pkl')

        dataset_dict = torch.load(args.output_dir + 'processed_data.pkl')
        noisy_train_data = dataset_dict['train']
        noisy_test_data = dataset_dict['test']
        train_control_data = torch.load(args.output_dir + 'train_control_data.pkl')  # 新增：加载控制信号的训练数据
        test_control_data = torch.load(args.output_dir + 'test_control_data.pkl')  # 新增：加载控制信号的测试数据

    train_latent_mask = create_mask(args, train_latent_data.shape)
    test_latent_mask = create_mask(args, test_latent_data.shape)

    dataset_dict = {'train': noisy_train_data,
                    'test': noisy_test_data,
                    'raw_train': train_data,  
                    'raw_test': test_data,  
                    'train_control': train_control_data,  
                    'test_control': test_control_data,  
                    'train_latent': train_latent_data,  
                    'test_latent': test_latent_data}  

    grounding_data = {'train_latent': train_latent_data * train_latent_mask,
                      'train_latent_mask': train_latent_mask,
                      'test_latent': test_latent_data * test_latent_mask,
                      'test_latent_mask': test_latent_mask}

    torch.save(dataset_dict, args.output_dir + 'processed_data.pkl')
    torch.save(grounding_data, args.output_dir + 'grounding_data.pkl')

    args_dict = {'mask_rate': args.mask_rate, 'noise_std': args.noise_std, 'model': args.model}
    torch.save(args_dict, args.output_dir + 'data_args.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--output-dir', type=str)
    parser.add_argument('--seq-len', type=int)
    parser.add_argument('--data-size', type=int)
    parser.add_argument('--delta-t', '-dt', type=float)
    parser.add_argument('--noise-std', type=float)
    parser.add_argument('--mask-rate', type=float, default=0.01)
    parser.add_argument('--model', choices=['pendulum'], default='pendulum')
    parser.add_argument('--change-only-mask-rate', action='store_true')
    parser.add_argument('--friction', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=12)

    args = parser.parse_args()

    if args.model == 'pendulum':
        args.create_raw_data = create_pendulum_data

    args = load_data_config(args)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    set_seed(args.seed)
    make_dataset(args)

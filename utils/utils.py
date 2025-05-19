import numpy as np
import torch
import random
from utils import ODE_with_control_dataset
import torch.nn as nn

# Reference: https://github.com/orilinial/GOKU
def reverse_sequences_torch(mini_batch):
    reversed_mini_batch = torch.zeros_like(mini_batch)
    T = mini_batch.size(1)
    device = mini_batch.device
    
    for b in range(mini_batch.size(0)):
        time_slice = np.arange(T - 1, -1, -1)
        # Convert to tensor and move to the same device as mini_batch
        time_slice_tensor = torch.tensor(time_slice, dtype=torch.long, device=device)
        
        reversed_sequence = torch.index_select(mini_batch[b, :, :], 0, time_slice_tensor)
        reversed_mini_batch[b, 0:T, :] = reversed_sequence
        
    return reversed_mini_batch

def annealing_factor_sched(start_af, end_af, ae, epoch, which_mini_batch, num_mini_batches):
    """
    :param start_af: start value of annealing factor 
    :param end_af: end value of annealing factor
    :param ae: annealing epochs - after these epochs, the annealing factor is set to the end value.
    :param epoch: current epoch
    :param which_mini_batch: current number of mini batch
    :param num_mini_batches: amount of mini batches
    :return: annealing factor to be used
    """
    if ae > 0:
        if epoch < ae:
            # compute the KL annealing factor appropriate for the current mini-batch in the current epoch
            annealing_factor = start_af + (end_af - start_af) * \
                                        (float(which_mini_batch + epoch * num_mini_batches + 1) /
                                         float(ae * num_mini_batches))
        else:
            annealing_factor = end_af
    else:
        # by default the annealing factor is unity
        annealing_factor = 1.0
    return annealing_factor


def set_seed(seed, fully_deterministic=True):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if fully_deterministic:
            torch.backends.cudnn.deterministic = True

def kld_gauss(mean_1, std_1, mean_2, std_2, masks=None, sum=False):
    EPS = torch.finfo(torch.float).eps
    """Using std to compute KLD"""
    kld_element =  (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
        (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
        std_2.pow(2) - 1)
    if masks is None:
        if sum:
            loss = 0.5 * torch.sum(kld_element.flatten(1), dim=1)
        else:
            loss = 0.5 * torch.mean(kld_element.flatten(1), dim=1)
    else:
        masks =(masks.sum(-1)>0)
        masks_sum = masks.sum(-1)
        masks_sum[masks_sum==0] = 1
        if sum:
            loss = 0.5 * torch.sum((kld_element * masks[..., None]).sum(-1), dim=-1)
        else:
            loss =	0.5 * torch.sum((kld_element * masks[...,None]).mean(-1), dim=-1) / masks_sum
    return loss

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def create_transforms(args):
    data_norm_params = torch.load(args.data_path + 'data_norm_params.pkl')
    data_transforms = {}
    # Normalization transformation
    if args.norm is not None:
        if args.norm == "zscore":
            # normalize_transform = ODE_dataset.NormalizeZScore(data_norm_params)
            normalize_transform = ODE_with_control_dataset.NormalizeZScore(data_norm_params)
        elif args.norm == "zero_to_one":
            # normalize_transform = ODE_dataset.NormalizeToUnitSegment(data_norm_params)
            normalize_transform = ODE_with_control_dataset.NormalizeToUnitSegment(data_norm_params)
        else:
            raise Exception("Choose valid normalization function: zscore or zero_to_one")
        data_transforms["normalize"] = normalize_transform
    return data_transforms


class StatesToSamples(nn.Module):
    def __init__(self, sample_dim, hidden_dim, state_dim):
        super(StatesToSamples, self).__init__()
        self.sample_dim = sample_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sample_dim)
        )

    def forward(self, data):
        out = self.net(data)
        return out
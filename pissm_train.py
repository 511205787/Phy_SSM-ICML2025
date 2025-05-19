import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import numpy as np
import torch
import torch.optim as optim
from utils import utils
from utils.Base_dataset import create_datasets
import models
import os
from config import load_pissm_train_config
import torch.nn.functional as F
from tqdm.auto import tqdm
import wandb
torch.autograd.set_detect_anomaly(True)

def setup_optimizer(model, lr, weight_decay, patience):
    # All parameters in the model
    all_parameters = list(model.parameters())
    
    # General parameters don't contain the special _optim key
    params = [p for p in all_parameters if not hasattr(p, "_optim")]

    # Create an optimizer with the general parameters
    optimizer = optim.AdamW(
        params, 
        lr=lr, 
        weight_decay=weight_decay,
    )

    # Add parameters with special hyperparameters
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [
        dict(s) for s in set(frozenset(hp.items()) for hp in hps)
    ]  # Unique dicts
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group(
            {"params": params, **hp}
        )

    # Create a lr scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.2)
    
    # Print optimizer info 
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([
            f"Optimizer group {i}",
            f"{len(g['params'])} tensors",
        ] + [f"{k} {v}" for k, v in group_hps.items()]))

    return optimizer, scheduler

def test_pissm(args, model, test_dataloader, device):
    model.eval()
    test_MAE_loss = 0
    Interp_MAE = 0
    Extra_MAE = 0
    MSE_loss_total = 0
    MSE_loss_interp = 0
    MSE_loss_extra = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(test_dataloader))
        for i_batch, test_batch in pbar:
            noisy_data, control_data, latent_data, target_data, t_arr = test_batch
            noisy_data = noisy_data.to(device)
            control_data = control_data.to(device)
            latent_data = latent_data.to(device)
            target_data = target_data.to(device)
            t_arr = t_arr.to(device)
            total_len = target_data.size(1)
            predicted_batch, predicted_z, _, _, _ = model(noisy_data[:, :args.seq_len], control_data, t=t_arr)

            extra_mae_loss = F.l1_loss(predicted_batch[:, args.seq_len:, :], target_data[:, args.seq_len:, :])
            mae_loss = F.l1_loss(predicted_batch, target_data)
            Interp_mae_loss = F.l1_loss(predicted_batch[:, :args.seq_len, :], target_data[:, :args.seq_len, :])
            mse_loss_total = F.mse_loss(predicted_batch, target_data)
            mse_loss_interp = F.mse_loss(predicted_batch[:, :args.seq_len, :], target_data[:, :args.seq_len, :])
            mse_loss_extra = F.mse_loss(predicted_batch[:, args.seq_len:, :], target_data[:, args.seq_len:, :])

            Interp_MAE += Interp_mae_loss.item()
            Extra_MAE += extra_mae_loss.item()
            test_MAE_loss += mae_loss.item()
            MSE_loss_total += mse_loss_total.item()
            MSE_loss_interp += mse_loss_interp.item()
            MSE_loss_extra += mse_loss_extra.item()
            break

        # Statistics
        pbar.set_description(
        'Batch Idx: (%d/%d) | MAE: %.3f' % 
        (i_batch, len(test_dataloader), test_MAE_loss/(i_batch+1)))
        wandb.log({'Test MAE': test_MAE_loss/(i_batch+1), 'Interp_MAE': Interp_MAE/(i_batch+1), 'Extra_MAE': Extra_MAE/(i_batch+1),
                   'Test MSE': MSE_loss_total/(i_batch+1), 'Interp_MSE': MSE_loss_interp/(i_batch+1), 'Extra_MSE': MSE_loss_extra/(i_batch+1),
                  })
    model.train()
    return Extra_MAE

def test_pissm_local(args, model, test_dataloader, device):
    model.eval()
    test_MAE_loss = 0
    Interp_MAE = 0
    Extra_MAE = 0
    MSE_loss_total = 0
    MSE_loss_interp = 0
    MSE_loss_extra = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(test_dataloader))
        for i_batch, test_batch in pbar:
            noisy_data, control_data, latent_data, target_data, t_arr = test_batch
            noisy_data = noisy_data.to(device)
            control_data = control_data.to(device)
            latent_data = latent_data.to(device)
            target_data = target_data.to(device)
            t_arr = t_arr.to(device)
            total_len = target_data.size(1)
            predicted_batch, predicted_z, _, _, _ = model(noisy_data[:, :args.seq_len], control_data, t=t_arr)

            extra_mae_loss = F.l1_loss(predicted_batch[:, args.seq_len:, :], target_data[:, args.seq_len:, :])
            mae_loss = F.l1_loss(predicted_batch, target_data)
            Interp_mae_loss = F.l1_loss(predicted_batch[:, :args.seq_len, :], target_data[:, :args.seq_len, :])
            mse_loss_total = F.mse_loss(predicted_batch, target_data)
            mse_loss_interp = F.mse_loss(predicted_batch[:, :args.seq_len, :], target_data[:, :args.seq_len, :])
            mse_loss_extra = F.mse_loss(predicted_batch[:, args.seq_len:, :], target_data[:, args.seq_len:, :])

            Interp_MAE += Interp_mae_loss.item()
            Extra_MAE += extra_mae_loss.item()
            test_MAE_loss += mae_loss.item()
            MSE_loss_total += mse_loss_total.item()
            MSE_loss_interp += mse_loss_interp.item()
            MSE_loss_extra += mse_loss_extra.item()

        # Statistics
        pbar.set_description(
        'Batch Idx: (%d/%d) | MAE: %.3f' % 
        (i_batch, len(test_dataloader), test_MAE_loss/(i_batch+1)))
        wandb.log({'Test MAE': test_MAE_loss/(i_batch+1), 'Interp_MAE': Interp_MAE/(i_batch+1), 'Extra_MAE': Extra_MAE/(i_batch+1),
                   'Test MSE': MSE_loss_total/(i_batch+1), 'Interp_MSE': MSE_loss_interp/(i_batch+1), 'Extra_MSE': MSE_loss_extra/(i_batch+1),
                  })
    model.train()
    return Extra_MAE

def validate_pissm(args, model, val_dataloader, device):
    model.eval()
    eval_MAE_loss = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(val_dataloader))
        for i_batch, val_batch in pbar:
            noisy_data, control_data, latent_data, target_data, t_arr = val_batch
            noisy_data = noisy_data.to(device)
            control_data = control_data.to(device)
            latent_data = latent_data.to(device)
            target_data = target_data.to(device)
            t_arr = t_arr.to(device)
            total_len = target_data.size(1)
            predicted_batch, _, _, _, _ = model(noisy_data[:, :args.seq_len], control_data, t=t_arr)
            extra_mae_loss = F.l1_loss(predicted_batch[:, args.seq_len:, :], target_data[:, args.seq_len:, :])
            mae_loss = F.l1_loss(predicted_batch, target_data)
            Interp_mae_loss = F.l1_loss(predicted_batch[:, :args.seq_len, :], target_data[:, :args.seq_len, :])
            
            eval_MAE_loss += mae_loss.item()
            break

        # Statistics
        pbar.set_description(
        'Batch Idx: (%d/%d) | MAE: %.3f' % 
        (i_batch, len(val_dataloader), eval_MAE_loss/(i_batch+1)))
        wandb.log({'Valid MAE': eval_MAE_loss/(i_batch+1)})
    model.train()
    return eval_MAE_loss


def train(args):
    # General settings
    utils.set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if not args.cpu else torch.device('cpu')
    ds_train, ds_val, ds_test = create_datasets(args)
    train_dataloader = torch.utils.data.DataLoader(ds_train, batch_size=args.mini_batch_size)
    val_dataloader = torch.utils.data.DataLoader(ds_val, batch_size=args.mini_batch_size)
    test_dataloader = torch.utils.data.DataLoader(ds_test, batch_size=args.mini_batch_size)

    # Create model - see models/PISSM.py for options
    model = models.__dict__["create_pissm_" + args.model]().to(device)
    print('Model: PISSM - %s created with %d parameters.' % (args.model, sum(p.numel() for p in model.parameters())))
    # L1 error on validation set (not test set!) for early stopping
    best_model = models.__dict__["create_pissm_" + args.model]().to(device)
    best_val_loss = np.inf

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint_dir = './checkpoints/' + args.model + '/PISSM/'  # Use your actual path
        checkpoint_path = os.path.join(checkpoint_dir, 'pissm_model_bt.pkl')
        assert os.path.isfile(checkpoint_path), f'Error: no checkpoint file found at {checkpoint_path}!'
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        print('Successfully loaded checkpoint from:', checkpoint_path)
    # Create optimizer here
    optimizer, scheduler = setup_optimizer(model, lr=args.learning_rate, 
                                           weight_decay=args.weight_decay, patience=args.patience)
    # Use wandb to log results
    wandb.init(project='Physics-informed SSM', name=f'PISSM-{args.learning_rate}-{args.model}-seed{args.seed}', config=args)
    if args.evaluate_only:
        test_loss = test_pissm_local(args, model, test_dataloader, device)
        return
    
    for epoch in range(args.num_epochs):
        epoch_loss_array = []
        train_mse_loss = 0
        train_loss = 0
        pbar = tqdm(enumerate(train_dataloader))
        for i_batch, mini_batch in pbar:
            noisy_data, control_data, latent_data, target_data, t_arr = mini_batch
            noisy_data = noisy_data.to(device)
            control_data = control_data.to(device)
            latent_data = latent_data.to(device)
            target_data = target_data.to(device)
            t_arr = t_arr.to(device)

            # Forward step
            pred_x, z_obs, z_generate, z_post, z_prior = model(noisy_data[:, :args.seq_len], control_data, t=t_arr)
            # Calculate loss:
            post_mean, post_std = z_post
            prior_mean, prior_std = z_prior
            # Reconstruction MSE loss
            if args.model == 'pendulum_friction':
                mse_loss = ((pred_x  - noisy_data) ** 2).mean((0, 1)).sum()
            else:
                mse_loss = ((pred_x - target_data) ** 2).mean((0, 1)).sum()
            loss_constraint = F.mse_loss(z_obs, z_generate)
            kl_annealing_factor = utils.annealing_factor_sched(args.kl_start_af, args.kl_end_af,
                                                               args.kl_annealing_epochs, epoch, i_batch,
                                                               len(train_dataloader))
            analytic_kl = utils.kld_gauss(post_mean, torch.exp(post_std), prior_mean, torch.exp(prior_std)).mean()
            if args.model == 'SIR':
                loss = mse_loss + 1e-5 * loss_constraint + 1e-5 * analytic_kl
            elif args.model == 'drone':
                loss = mse_loss + 100 * loss_constraint + analytic_kl
            else:
                loss = mse_loss + kl_annealing_factor * (analytic_kl) + loss_constraint
            # Backward step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_mse_loss += mse_loss.item()
            train_loss += loss.item()

            # Statistics
            epoch_loss_array.append(loss.item())
            pbar.set_description(
            'Epoch: (%d) | Batch Idx: (%d/%d) | Loss: %.3f | MSE: %.3f' % 
            (epoch, i_batch, len(train_dataloader), train_loss/(i_batch+1), train_mse_loss/(i_batch+1))
            )
        wandb.log({'Train_total_loss': train_loss/(i_batch + 1), 'Train MSE': train_mse_loss/(i_batch+1)})
        # Calculate validation loss
        best_model.load_state_dict(model.state_dict())
        val_loss = validate_pissm(args, best_model, val_dataloader, device)
        # Test extrapolation MAE here
        if epoch % args.saving_epoch == 0:
            test_loss = test_pissm(args, model, test_dataloader, device)

            log_dict = {"args": args,
                        "model": model.state_dict()
                        }
            save_path = os.path.join(args.checkpoints_dir, f'pissm_model_epoch_{epoch}.pkl')
            torch.save(log_dict, save_path)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model.load_state_dict(model.state_dict())

    # Save model and run hyper parameters
    log_dict = {"args": args,
                "model": best_model.state_dict()
                }
    torch.save(log_dict, args.checkpoints_dir + 'pissm_model_bt.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    # Run parameters
    parser.add_argument('-n', '--num-epochs', type=int)
    parser.add_argument('-mbs', '--mini-batch-size', type=int)
    parser.add_argument('--seed', type=int, default=1)

    # Data parameters
    parser.add_argument('-sl', '--seq-len', type=int)
    parser.add_argument('--delta-t', type=float)
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--norm', type=str, choices=['zscore', 'zero_to_one'], default=None)

    # Optimizer parameters
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight-decay', type=float, default=0.01)
    parser.add_argument('--patience', default=100, type=float, help='Patience for learning rate scheduler')
    
    # Model parameters
    parser.add_argument('--model', type=str, choices=['pendulum_friction', 'SIR', 'drone'],
                        default='SIR')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout for State space Model')
    parser.add_argument('--saving_epoch', type=int)
    
    # KL Annealing factor parameters
    parser.add_argument('--kl-annealing-epochs', type=int)
    parser.add_argument('--kl-start-af', type=float)
    parser.add_argument('--kl-end-af', type=float)
    # SIR countries
    parser.add_argument('--train_countries', nargs='+', help='List of countries for training')
    parser.add_argument('--val_countries', nargs='+', help='List of countries for validation')
    parser.add_argument('--test_countries', nargs='+', help='List of countries for testing')

    parser.add_argument('--checkpoints-dir', type=str, default='checkpoints/')
    parser.add_argument('--cpu', action='store_true')
    # General
    parser.add_argument('--resume', '-r', type=bool, default=False, help='Resume from checkpoint')
    parser.add_argument('--evaluate-only', type=bool, default=False, help='Only evaluate the model and plot the figure locally')
    args = parser.parse_args()
    args = load_pissm_train_config(args)
    args.checkpoints_dir = args.checkpoints_dir + args.model + '/PISSM'+ '/'
    if not os.path.exists(args.checkpoints_dir):
        os.makedirs(args.checkpoints_dir)
    train(args)





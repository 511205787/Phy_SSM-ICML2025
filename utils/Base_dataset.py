
import os
from utils import utils
from utils.ODE_with_control_dataset import ODEDataSet_with_control
from utils.SIR_dataset import SIRDataSet
from utils.Drone_dataset import DroneFlightDataset

def create_datasets(args):
    """
    Create and return train/val/test datasets based on args.model.
    Returns:
        ds_train: training dataset
        ds_val:   validation dataset
        ds_test:  test dataset
    """

    # 1) Prepare any required data transforms
    if args.model == 'SIR':
        # no extra transforms needed for SIR, use default normalization inside the dataset
        data_transforms = True
    elif args.model == 'pendulum_friction':
        # custom transforms for pendulum with friction
        data_transforms = utils.create_transforms(args)
    else:
        # other models don't use transforms
        data_transforms = None

    # 2) Build SIR datasets
    if args.model == 'SIR':
        # First dataset computes mean/std internally (using all_countries)
        ds_all = SIRDataSet(
            data_dir=args.data_path,
            countries=args.all_countries,
            calculate_stats=True
        )
        ds_train = SIRDataSet(
            data_dir=args.data_path,
            countries=args.train_countries,
            mean=ds_all.mean, std=ds_all.std,
            training_mode=True, calculate_stats=False
        )
        ds_val = SIRDataSet(
            data_dir=args.data_path,
            countries=args.val_countries,
            mean=ds_all.mean, std=ds_all.std,
            training_mode=False, calculate_stats=False
        )
        ds_test = SIRDataSet(
            data_dir=args.data_path,
            countries=args.test_countries,
            mean=ds_all.mean, std=ds_all.std,
            training_mode=False, calculate_stats=False
        )
        return ds_train, ds_val, ds_test

    # 3) Build drone flight datasets
    elif args.model == 'drone':
        # Train set computes stats and feature indices
        ds_train = DroneFlightDataset(
            data_dir=args.data_path,
            ds_type='train',
            seq_len=args.seq_len + 200,
            discard_fraction=0
        )
        stats = ds_train.get_stats()
        feature_indices = ds_train.get_feature_indices()

        # Apply same stats/indices to val and test
        ds_val = DroneFlightDataset(
            data_dir=args.data_path,
            ds_type='valid',
            seq_len=args.seq_len + 200,
            discard_fraction=0,
            stats=stats,
            feature_indices=feature_indices
        )
        ds_test = DroneFlightDataset(
            data_dir=args.data_path,
            ds_type='test',
            seq_len=args.seq_len + 200,
            discard_fraction=0,
            stats=stats,
            feature_indices=feature_indices
        )
        return ds_train, ds_val, ds_test

    # 4) Build ODE-with-control (including pendulum_friction)
    else:
        file_path = os.path.join(args.data_path, 'processed_data.pkl')
        ds_train = ODEDataSet_with_control(
            file_path=file_path,
            ds_type='train',
            seq_len=args.seq_len + 80,
            random_start=False,
            transforms=data_transforms
        )
        ds_val = ODEDataSet_with_control(
            file_path=file_path,
            ds_type='val',
            seq_len=args.seq_len + 80,
            random_start=False,
            transforms=data_transforms
        )
        ds_test = ODEDataSet_with_control(
            file_path=file_path,
            ds_type='test',
            seq_len=args.seq_len + 80,
            random_start=False,
            transforms=data_transforms
        )
        return ds_train, ds_val, ds_test

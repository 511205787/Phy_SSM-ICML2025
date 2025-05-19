def load_data_config(args):
    if args.model == 'pendulum':
        args.output_dir = 'data/pendulum/' if not args.friction else 'data/pendulum_friction/'
        args.seq_len = 300
        args.data_size = 500
        args.delta_t = 0.05
        args.noise_std = 0.3
        args.seed = 13
    return args

def load_pissm_train_config(args):
    if args.model == 'pendulum_friction':
        args.num_epochs = 500
        args.mini_batch_size = 64
        args.seq_len = 160
        args.delta_t = 0.05
        args.data_path = 'data/pendulum_friction/'
        args.norm = 'zero_to_one'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001
        args.grounding_loss = 1000.0
        args.saving_epoch = 10

    if args.model == 'drone':
        args.num_epochs = 20
        args.mini_batch_size = 64
        args.seq_len = 801
        args.data_path = 'data/drone_data/'
        args.model = 'drone'
        args.kl_annealing_epochs = 1
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001
        args.grounding_loss = 0.0
        args.saving_epoch = 1
        args.learning_rate = 1e-4

    if args.model == 'SIR':
        args.num_epochs = 400
        args.mini_batch_size = 32
        args.seq_len = 160 # 
        args.data_path = 'data/sir_data/'
        args.kl_annealing_epochs = 200
        args.kl_start_af = 0.00001
        args.kl_end_af = 0.00001
        args.grounding_loss = 0.0
        args.all_countries = ['Armenia', 'Brazil', 'France', 'Germany', 'Gabon', 'Ireland', 'Spain', 'UnitedKingdom',]
        args.train_countries = ['Armenia', 'Brazil', 'France', 'Germany', 'Gabon']
        args.val_countries = ['UnitedKingdom']
        args.test_countries = ['Ireland', 'Spain']
        args.saving_epoch = 10
        args.learning_rate = 1e-3
    args.seed = 14 #
    return args
import numpy as np
import gym
import skimage.transform
from tqdm import trange


def get_theta(obs):
    """Transforms coordinate basis from the defaults of the gym pendulum env."""
    theta = np.arctan2(obs[0], -obs[1])
    theta = theta + np.pi/2
    theta = theta + 2*np.pi if theta < -np.pi else theta
    theta = theta - 2*np.pi if theta > np.pi else theta
    return theta


def preproc(X, side):
    """Crops, downsamples, desaturates, etc. the rgb pendulum observation."""
    X = X[..., 0][220:-110, 165:-165] - X[..., 1][220:-110, 165:-165]    
    # X = X[..., 0][100:-100, 100:-100] - X[..., 1][100:-100, 100:-100]
    return skimage.transform.resize(X, [int(side), side]) / 255.


def step_env(args, env, u, params, additional_params):
    th, thdot = env.state

    g = 10.0
    m = 1.0
    b = additional_params['b']
    l = params["l"]
    dt = env.dt

    if args.friction:
        newthdot = thdot + ((-g/l) * np.sin(th + np.pi) - (b/m) * thdot + (u)/(m * l**2)) * dt
    else:
        newthdot = thdot + ((-g/l) * np.sin(th + np.pi) + u/(m * l**2)) * dt

    newth = th + newthdot*dt
    newthdot = np.clip(newthdot, -env.max_speed, env.max_speed)

    env.state = np.array([newth, newthdot])
    return env._get_obs()


def get_params():
    l = np.random.uniform(1.0, 2.0)
    torque_coeff = np.random.uniform(-5.0, 5.0)
    # torque_coeff = 6.0
    params = {"l": l, "torque_coeff": torque_coeff}
    return params


def get_unlearned_params():
    b = 0.7
    params = {'b': b}
    return params


def reset_env(env, min_angle=0., max_angle=np.pi / 6):
    angle_ok = False
    while not angle_ok:
        obs = env.reset()
        theta_init = np.abs(get_theta(obs))
        if min_angle < theta_init < max_angle:
            angle_ok = True
    return obs  # Return the observation


def create_pendulum_data(args, side=28):
    env_name = 'Pendulum-v0'
    env = gym.make(env_name).unwrapped
    env.seed(args.seed)
    data = np.zeros((args.data_size, args.seq_len, side, side))
    control_data = np.zeros((args.data_size, args.seq_len))
    latent_data = np.zeros((args.data_size, args.seq_len, 2))
    params_data = []

    for trial in trange(args.data_size):
        obs = reset_env(env)
        params = get_params()
        unlearned_params = get_unlearned_params()

        for step in range(args.seq_len):
            processed_frame = preproc(env.render('rgb_array'), side)
            data[trial, step] = processed_frame

            t = step * (env.dt)  # Current time
            omega = obs[-1]  # Angular velocity θ_dot
            torque_coeff = params["torque_coeff"]
            torque = (torque_coeff) * np.cos(2 * np.pi * 0.1 * t)
            # torque = torque_coeff * np.sin(2 * np.pi * 0.1 * t)
            # torque = np.random.uniform(-10.0, 10.0)  # random apply torque
            # obs = step_env(args, env, [0.], params, unlearned_params)
            obs = step_env(args, env, torque, params, unlearned_params)

            latent_data[trial, step, 0] = get_theta(obs)
            latent_data[trial, step, 1] = obs[-1]
            control_data[trial, step] = torque 

        params_data.append(params)

    env.close()
    return data, control_data, latent_data, params_data


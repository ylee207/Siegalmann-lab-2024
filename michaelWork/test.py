import os
import time
import argparse
import pickle
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.buffers import ReplayBuffer

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.AtoBAviary import AtoBAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin')  # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm')  # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False

INIT = np.array([[0, 1, 1] for i in range(1)])
TARGET_RWRD = 3000
LENGTH = 5e7
DETER = True

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):
    print("TRGT = ", TARGET_RWRD, ", LENGTH = ", LENGTH, ", INIT = ", INIT, ", deterministic = ", DETER)
    filename = os.path.join(output_folder, 'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    train_env = make_vec_env(AtoBAviary,
                             env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT),
                             n_envs=1,
                             seed=0)
    eval_env = AtoBAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT)

    # Initialize replay buffer
    replay_buffer = ReplayBuffer(buffer_size=1000000, observation_space=train_env.observation_space, action_space=train_env.action_space)

    # Check the environment's spaces
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)

    # Train the model
    model = PPO('MlpPolicy',
                train_env,
                verbose=1,
                policy_kwargs=dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])],
                                   activation_fn=torch.nn.ReLU,
                                   log_std_init=-2.0,
                                   ortho_init=True,
                                   normalize_images=True,
                                   optimizer_class=torch.optim.AdamW,
                                   optimizer_kwargs=dict(weight_decay=0.01)))

    # Target cumulative rewards (problem-dependent)
    target_reward = TARGET_RWRD
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 best_model_save_path=filename+'/',
                                 log_path=filename+'/',
                                 eval_freq=int(1000),
                                 deterministic=DETER,
                                 render=False)
    model.learn(total_timesteps=int(LENGTH) if local else int(1e2),  # shorter training in GitHub Actions pytest
                callback=eval_callback,
                log_interval=100)

    # Save the model and replay buffer
    model.save(filename+'/final_model.zip')
    with open(filename+'/replay_buffer.pkl', 'wb') as f:
        pickle.dump(replay_buffer, f)
    print(filename)

    # Print training progression
    with np.load(filename+'/evaluations.npz') as data:
        for j in range(data['timesteps'].shape[0]):
            print(str(data['timesteps'][j])+","+str(data['results'][j][0]))

    if local:
        input("Press Enter to continue...")

    # Load the best model
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
        return
    model = PPO.load(path)

    # Show (and record a video of) the model's performance
    test_env = AtoBAviary(gui=gui,
                          obs=DEFAULT_OBS,
                          act=DEFAULT_ACT,
                          initial_xyzs=INIT,
                          record=record_video)
    test_env_nogui = AtoBAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, initial_xyzs=INIT)

    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                    num_drones=DEFAULT_AGENTS if multiagent else 1,
                    output_folder=output_folder,
                    colab=colab)

    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10)
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n")

    obs, info = test_env.reset(seed=42, options={})
    start = time.time()
    for i in range((test_env.EPISODE_LEN_SEC+2)*test_env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=DETER)
        obs, reward, terminated, truncated, info = test_env.step(action)
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
        if DEFAULT_OBS == ObservationType.KIN:
            if not multiagent:
                logger.log(drone=0,
                           timestamp=i/test_env.CTRL_FREQ,
                           state=np.hstack([obs2[0:3], np.zeros(4), obs2[3:15], act2]),
                           control=np.zeros(12))
            else:
                for d in range(DEFAULT_AGENTS):
                    logger.log(drone=d,
                               timestamp=i/test_env.CTRL_FREQ,
                               state=np.hstack([obs2[d][0:3], np.zeros(4), obs2[d][3:15], act2[d]]),
                               control=np.zeros(12))
        test_env.render()
        print(terminated)
        sync(i, start, test_env.CTRL_TIMESTEP)
        if terminated:
            obs = test_env.reset(seed=42, options={})
    test_env.close()

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    # Define and parse (optional) arguments for the script
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent', default=DEFAULT_MA, type=str2bool, help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool, help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video', default=DEFAULT_RECORD_VIDEO, type=str2bool, help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str, help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool, help='Whether example is being run by a notebook (default: "False")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))

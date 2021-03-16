
import gym
import roboschool
from gym import wrappers


import joblib
import numpy as np
import os, sys, subprocess
from copy import deepcopy


from play_pong_train import INF,SEED
# from stable_baselines import PPO1
from replace.pposgd_simple import PPO1
from stable_baselines.common.vec_env import DummyVecEnv#,SubprocVecEnv, VecVideoRecorder
# import pickle as pkl



STEPS = INF
def play(env, model, player_n, steps, opponent):
    obs_list = []
    acts_list = []
    rwds_list = []
    dones_list = []
    hidden_list = []
    trajectory_dic = None
    while 1:
        obs = env.reset()
        count = 0
        while 1:
            # try:
            count += 1
            old_obs = deepcopy(obs)
            
            if '2017' in opponent:
                a = model.act(obs)
                lastpi = a
            else:
                a, _ ,lastpi = model.predict(obs)
            
            obs, rew, done, info = env.step(a)
            
            hidden_list.append(lastpi.copy())
            acts_list.append(a.copy())
            obs_list.append(old_obs.copy())
            rwds_list.append(rew.copy())
            dones_list.append(done.copy())
            
            if count == steps:
                print("player{} finished!".format(player_n),np.vstack(hidden_list).shape)
                # dones_list[-1] = [True]
                trajectory_dic = {
                    'all_hidden':np.vstack(hidden_list),
                    "all_obvs":np.vstack(obs_list),
                    "all_acts":np.vstack(acts_list),
                    "all_rwds":np.vstack(rwds_list),
                    "all_dones":np.vstack(dones_list)
                }
                break

        break
    return trajectory_dic

def test(game_server_guid, opponent, save_oppo_traj):
    env = gym.make("RoboschoolPong-v1")
    env.seed(SEED)
    player_n = 1
    env.unwrapped.multiplayer(env, game_server_guid=game_server_guid, player_n=player_n)
    # print(">>>>game_server_id[player1]:",game_server_id)
    env = DummyVecEnv([lambda: env])
    print("player 2 running in Env {0}".format(game_server_guid))
    

    if opponent=="2017may1":
        from pretrained.RoboschoolPong_v0_2017may1 import SmallReactivePolicy as Pol1
        model = Pol1(env.observation_space, env.action_space)
    else:
        assert False,"Wrong opponent name: {}".format(opponent)

    trajectory_dic = play(env, model, player_n, STEPS, opponent)
    if save_oppo_traj != "":
        joblib.dump(trajectory_dic, save_oppo_traj)
    


if sys.argv[2] == "test":
    test(sys.argv[1], sys.argv[3], sys.argv[4])

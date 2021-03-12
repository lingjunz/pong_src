from play_pong_train import INF,SEED
import os, sys, subprocess
import numpy as np
import gym
from gym import wrappers
import roboschool
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv, VecVideoRecorder
import pickle as pkl
from copy import deepcopy

STEPS = INF
def play(env, model, player_n,steps):
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

def test():
    env = gym.make("RoboschoolPong-v1")
    env.seed(SEED)
    player_n = 1
    env.unwrapped.multiplayer(env, game_server_guid=sys.argv[1], player_n=player_n)
    env = DummyVecEnv([lambda: env])
    print("player 2 running in Env {0}".format(sys.argv[1]))

    # from RoboschoolPong_v0_2017may1 import SmallReactivePolicy as Pol1
    # pi = Pol1(env.observation_space, env.action_space)
    from stable_baselines import PPO1
    import joblib
    modelpath = "./Log/ppo1TFS-02262021-114243-victim/model/best_model.pkl"
    model = PPO1.load(modelpath,env=env)

    savepath = "./Log/ppo1TFS-02262021-114243-victim/model/ppo1-advtrain-trajectory-test.data"
    trajectory_dic = play(env, model, player_n, STEPS)

    joblib.dump(trajectory_dic,savepath)
    
    # with open(savepath, 'wb+') as f:
    #     pkl.dump(trajectory_dic, f, protocol=2)


if sys.argv[2] == "test":
    test()

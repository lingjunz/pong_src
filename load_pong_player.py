
import sys,os
import numpy as np
import joblib
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1
from pposgd_wrap import PPO1_model_value
import gym
import roboschool

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def play(env, model, player_n, steps):
    obs_list = []
    acts_list = []
    rwds_list = []
    dones_list = []
    hidden_list = []
    trajectory_dic = None
    while True:
        obs = env.reset()
        count = 0
        while True:
            
            count += 1
            old_obs = obs.copy()
            a, _ ,lastpi = model.predict(obs)
            obs, rew, done, info = env.step(a)
            # print("player:{}, steps:{}".format(player_n,count))
            hidden_list.append(lastpi.copy())
            acts_list.append(a.copy())
            obs_list.append(old_obs.copy())
            rwds_list.append(rew.copy())
            dones_list.append(done.copy())

            if count == steps:
                print("player{} finished!".format(player_n))
                # dones_list[-1] = [True]
                trajectory_dic = {
                    'all_hidden':np.vstack(hidden_list),
                    "all_obvs":np.vstack(obs_list),
                    "all_acts":np.vstack(acts_list),
                    "all_rwds":np.vstack(rwds_list),
                    "all_dones":np.vstack(dones_list)
                }
                break
            
            if count%10000==0:
                print(count, info)

        break
    return trajectory_dic

            
def test(game_server_id, seed, player_n, modelpath, save_traj, steps):
    env = gym.make("RoboschoolPong-v1")
    env.seed(seed)
    env.unwrapped.multiplayer(env, game_server_guid=game_server_id, player_n=player_n)
    env = DummyVecEnv([lambda: env])
    print("player {0} running in Env {1}".format(player_n+1, game_server_id))
    
    if os.path.exists(modelpath):

        if 'ppo1' in modelpath:
            model = PPO1.load(modelpath,env=env)
        elif 'pposgd' in modelpath:
            model = PPO1_model_value.load(modelpath,env=env)
        else:
            assert False,"wrong modelpath:{}".format(modelpath)
    else:
        assert False,"Pretrained model is not found in {}".format(modelpath)
        
    trajectory_dic = play(env, model, player_n, steps)
    if save_traj and player_n==0:
        savepath = modelpath[:-4]+"_ppoadv_player{}_traj.data".format(player_n)
        joblib.dump(trajectory_dic,savepath)
        print("Trajectories are saved in {} successfully".format(savepath))

        
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    # memo, hyper_index and server for create the serve
    parser.add_argument("--game_server_id", type=str)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--save_traj", type=bool, default=False)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--player_n", type=int, default=0)
    parser.add_argument("--modelpath", type=str)
    
    return parser.parse_args()        

if __name__=="__main__":
    
    args = parse_args()
    if args.mode == "test":
        print(args.steps)
        test( args.game_server_id ,args.seed, args.player_n, args.modelpath, args.save_traj, args.steps+1 )

    
    
    

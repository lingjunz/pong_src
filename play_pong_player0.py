
from play_pong_train import INF,SEED
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)

import os
import sys
import time
import joblib
import argparse
import traceback

from datetime import datetime
from shutil import copyfile
# import subprocess
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import gym
import roboschool
from ppoadv.policies import MlpPolicy_hua 
from ppoadv.pposgd_wrap import PPO1_model_value
# from stable_baselines import PPO1
from replace.pposgd_simple import PPO1
from replace.policies import MlpPolicy
from stable_baselines import logger
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor

from utils.logger import setup_logger
from utils.pong_utils import make_dirs,test

# import pickle as pkl
# import multiprocessing
# import pandas as pd
# global player_n
# player_n = 0
# import json
# USE_VIC = False


best_mean_reward = -np.inf
dir_dict = None
TRAINING_ITER = INF




def callback(_locals, _globals):
    def shift(arr, num, fill_value=np.nan):
        result = np.empty_like(arr)
        if num>0:
            result[:num] = fill_value
            result[num:] = arr[:-num]
        elif num<0:
            result[num:] = fill_value
            result[:num] = arr[-num:]
        else:
            result = arr
        return result

    global best_mean_reward
    # Evaluate policy training performance
    copyfile("/tmp/monitor/{0}/{1}/monitor.csv".format(dir_dict['_hyper_weights_index'],0),
             "{0}learn_monitor.csv".format(dir_dict['log'])
             )
    x, y = ts2xy(load_results(dir_dict['log']), 'timesteps')
    if len(x) > 0:
        mean_reward = np.mean((y[-100:]-shift(y[-100:],1,fill_value=0.0)))
        print(x[-1], 'timesteps')
        print(y[-100:])
        print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

        # New best model, you could save the agent here
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            # Example for saving best model
            print("Saving new best model")
            _locals['self'].save(dir_dict['model'] + 'best_model.pkl')
    return True

def advlearn(env, model_name=None, dir_dict=None):

    _, _ = setup_logger(SAVE_DIR, EXP_NAME)
    net_arch = [64,64,dict(pi=[6])]
    if model_name == 'ppo1Adv':
        ## inline hyperparameters
        ## param timesteps_per_actorbatch: timesteps per actor per update
        ## other inline hyperparameters is by default choice in file 'PPO1_model_value'
        model = PPO1_model_value(MlpPolicy_hua, env, 
                        timesteps_per_actorbatch=1000, verbose=1,
                         tensorboard_log=dir_dict['tb'], 
                         hyper_weights=dir_dict['_hyper_weights'],
                         benigned_model_file=None, full_tensorboard_log=False,
                         black_box_att=dir_dict['_black_box'], attention_weights=dir_dict['_attention'],
                         model_saved_loc=dir_dict['model'], 
                         clipped_attention=dir_dict['_clipped_attention'], 
                         exp_method=dir_dict['_x_method'], 
                         mimic_model_path=dir_dict['_mimic_model_path'],
                         save_victim_traj=dir_dict['_save_victim_traj'],
                         save_trajectory = dir_dict['_save_trajectory'],
                         policy_kwargs={"net_arch":net_arch})
    else:
        model = PPO1(MlpPolicy, env, timesteps_per_actorbatch=1000, verbose=1, save_trajectory = dir_dict['_save_trajectory'],
                     tensorboard_log=dir_dict['tb'], policy_kwargs={"net_arch":net_arch})
    try:
        model,trajectory_dic = model.learn(TRAINING_ITER, callback=callback, seed=dir_dict['_seed'])
        if dir_dict['_save_trajectory']==1:
            joblib.dump(trajectory_dic,"{}train-trajectory.data".format(dir_dict['model']))
        
    except ValueError as e:
        traceback.print_exc()
        print("Learn exit!")
    
    
    model_file_name = "{0}agent.pkl".format(dir_dict['model'])
    model.save(model_file_name)

def advtrain(server_id, model_name="ppo1", dir_dict=None):
    env = gym.make("RoboschoolPong-v1")
    env.seed(dir_dict['_seed'])
    # print(">>>>game_server_id[advtrain]:",server_id)
    env.unwrapped.multiplayer(env, game_server_guid=server_id, player_n=dir_dict["_player_index"])
    # Only support PPO2 and num cpu is 1

    if "ppo1" in model_name:
        n_cpu = 1
        env = SubprocVecEnv([lambda: env for i in range(n_cpu)])
    else:
        raise NotImplementedError
    advlearn(env, model_name=model_name, dir_dict=dir_dict)




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str)
    parser.add_argument("--server", type=str)
    parser.add_argument("--mod", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--hyper_index", type=int)
    parser.add_argument("--x_method", type=str)
    parser.add_argument("--mimic_model_path", type=str)
    parser.add_argument("--save_victim_traj", type=int)
    parser.add_argument("--save_trajectory", type=int)
    parser.add_argument("--seed", type=int)
    

    parser.add_argument("--player_index", type=int, default=0)
    parser.add_argument("--test_model_file", type=str, default=None)
    # parser.add_argument("--pretrain", action='store_true', default=False)

    return parser.parse_args()


if __name__=="__main__":

    now = datetime.now()
    date_time = now.strftime("%m%d%Y-%H%M%S")

    args = parse_args()
    # print(args)
    memo = args.memo
    server_id = args.server
    mode = args.mod
    model_name = args.model_name.lower()
    hyper_weights_index = args.hyper_index
    player_index = args.player_index
    test_model_file = args.test_model_file
    # hyper_weights = [0.0, -0.1, 0.0, 1, 0, 10, True, True, False] # blackbox
    hyper_weights = [0.0, -0.1, 0.0, 1, 0, 10, False, True, False] # whitebox

    
    dir_dict= {
        "tb": "Log/{}-{}/tb/".format(memo,date_time),
        "model": "Log/{}-{}/model/".format(memo, date_time),
        "log": "Log/{}-{}/".format(memo,date_time),
        "_hyper_weights": hyper_weights,
        "_hyper_weights_index": hyper_weights_index,
        "_video": False,
        "_player_index": player_index,
        "_test_model_file": test_model_file,

        ## whether black box or attention
        "_black_box": hyper_weights[-3],
        "_attention": hyper_weights[-2],
        "_clipped_attention": hyper_weights[-1],
        "_seed": args.seed,
        "_x_method": args.x_method,
        "_mimic_model_path": args.mimic_model_path,
        "_save_victim_traj": args.save_victim_traj,
        "_save_trajectory": args.save_trajectory
    }

    SAVE_DIR = './agent_zoo/'+ "Pong"
    EXP_NAME = str(args.seed)

    if "train" in mode :
        make_dirs(dir_dict)
        advtrain(server_id, model_name=model_name, dir_dict=dir_dict)

    elif "test" in mode:
        make_dirs(dir_dict)
        copyfile(dir_dict["_test_model_file"], 
                "{0}agent{1}.pkl".format(dir_dict['model'], dir_dict['_player_index']))
        test(server_id, model_name=model_name, seed=dir_dict['_seed'], dir_dict=dir_dict)

    else:
        assert False,"please set the correct mode:{}.".format(mode)
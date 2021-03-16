import numpy as np
import os
import gym
import roboschool
from ppoadv.pposgd_wrap import PPO1_model_value

def make_dirs(dir_dict):
    for key, value in dir_dict.items():
        if key.startswith("_"):
            continue
        os.makedirs(value, exist_ok=True)
    f = open("{}agent.config".format(dir_dict["log"]),"w")
    f.write( str(dir_dict) )
    f.close()

def play(env, model):
    while True:
        obs = env.reset()
        while True:
            action = model.predict(obs)[0]
            obs, rew, done, info = env.step(action)
            if done:
               break
        break

def test(server_id, model_name, seed, dir_dict=None):
    env = gym.make("RoboschoolPong-v1")
    env.seed(seed)
    env.unwrapped.multiplayer(env, game_server_guid=server_id, player_n=dir_dict["_player_index"])
    model_name = "{0}agent{1}.pkl".format(dir_dict['model'], dir_dict['_player_index'])

    model = PPO1_model_value.load(model_name, env=env, timesteps_per_actorbatch=3000, tensorboard_log=dir_dict['tb'])
    # load the pretrained_model 
    play(env, model)


def infer_obs_next_ph(obs_ph):
    '''
    This is self action at time t+1
    :param obs_ph:
    :return:
    '''
    obs_next_ph = np.zeros_like(obs_ph)
    obs_next_ph[:-1, :] = obs_ph[1:, :]
    return obs_next_ph


def infer_obs_opp_ph(obs_ph):
    '''
    This is oppos observation at time t
    :param obs_ph:
    :return:
    '''
    abs_opp_ph = np.zeros_like(obs_ph)
    neg_sign = [-1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, 1]
    abs_opp_ph[:, :4] = obs_ph[:, 4:8]
    abs_opp_ph[:, 4:8] = obs_ph[:, 0:4]
    abs_opp_ph[:, 8:] = obs_ph[:, 8:]
    return abs_opp_ph * neg_sign


def infer_action_previous_ph(action_ph):
    '''
    This is self action at time t-1
    :param obs_ph:
    :return:
    '''
    data_length = action_ph.shape[0]
    action_prev_ph = np.zeros_like(action_ph)
    action_prev_ph[1:, :] = action_ph[0:(data_length - 1), :]
    return action_prev_ph


def infer_obs_mask_ph(obs_ph):
    abs_mask_ph = np.zeros_like(obs_ph)
    abs_mask_ph[:, :4] = obs_ph[:, :4]
    abs_mask_ph[:, 8:] = obs_ph[:, 8:]
    return abs_mask_ph 

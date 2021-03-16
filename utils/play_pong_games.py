import os, sys, subprocess, time
import argparse
import roboschool
import multiplayer

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
\
# python play_pong_games.py --memo selfplay --server playGames --mod test --model_name ppo1_victim --hyper_index 9 --x_method None --mimic_model_path None --save_victim_traj True --seed 101

# python play_pong_games.py --memo advplay --server playGames --mod test --hyper_index 9 --save_victim_traj True --seed 101

# We assume the Game Server running forever
INF = 4000000

def parse_args():

    parser = argparse.ArgumentParser()

    # memo, hyper_index and server for create the serve
    parser.add_argument("--memo", type=str, default='ppo_pong')
    parser.add_argument("--server", type=str, default='pongdemo_adv')
    parser.add_argument("--mod", type=str, default="advtrain")

    # model_name (previous distinguish ppo2 and ppo1, now is ppo)
    # parser.add_argument("--model_name", type=str, default="ppo1_oppomodel")
    parser.add_argument("--hyper_index", type=int, default=3)
    # seed value
    # parser.add_argument("--x_method", type=str, default="grad")
    # parser.add_argument("--mimic_model_path", type=str, default="../pretrain/saved/mimic_model.h5")
    parser.add_argument("--save_victim_traj", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=2999)

    return parser.parse_args()

if __name__=="__main__":

    args = parse_args()

    memo = args.memo
    server_id = args.server
    mode = args.mod

    # model_name = args.model_name
    hyper_index = args.hyper_index


    # x_method = args.x_method
    # mimic_model_path = args.mimic_model_path
    save_victim_traj = args.save_victim_traj
    seed = args.seed

    # create the gameserver, the same as enviroment
    game_server_id = server_id+"{0}".format(hyper_index)
    game = roboschool.gym_pong.PongSceneMultiplayer()
    gameserver = multiplayer.SharedMemoryServer(game, game_server_id, want_test_window=False, profix=str(hyper_index))
    
    player_args =  "--game_server_id={0} --mode={1} " \
               "--seed={2} --save_traj={3} --steps={4}".format(game_server_id, mode, seed, save_victim_traj, INF)
    
    player_args = player_args.split(" ")
    sys_cmd = [sys.executable, 'load_pong_player.py']
    path0 = "./Log/ppo1TFS-02262021-114243-victim/model/best_model.pkl"
    path1 = "Log/pposgd-03052021-183609/model/best_model.pkl"
    sys_cmd0 = sys_cmd + player_args + ["--player_n=0", "--modelpath={}".format(path0)]
    sys_cmd1 = sys_cmd + player_args + ["--player_n=1", "--modelpath={}".format(path1)]

#     print(sys_cmd1)
    start_time = time.time()
    # setting up the player 0 & 1
    p0 = subprocess.Popen(sys_cmd0)
    
    p1 = subprocess.Popen(sys_cmd1)
    
    try:
        gameserver.serve_forever(INF)
    except ValueError:
        print("===>",time.time()-start_time)
        print("End of training!")
    

    
    





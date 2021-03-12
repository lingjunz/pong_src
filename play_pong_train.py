import os, sys, subprocess
import argparse
import roboschool, multiplayer
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# python play_pong_train.py --memo ppo1TFS --server ppodemo_train --mod advtrain --model_name ppo1Model --hyper_index 9 --x_method None --mimic_model_path None # train a model against 2017may1 with customized net_arch


# python play_pong_train.py --memo ppo1TFS --server ppodemo_train --mod advtrain --model_name ppo1Model --hyper_index 9 --x_method None --mimic_model_path None --save_victim_traj True # train a model against victim agent with customized net_arch

# python play_pong_train.py --memo pposgd --server ppoadv_train --mod advtrain --model_name ppo1_oppomodel --hyper_index 9 --x_method None --mimic_model_path None --save_victim_traj True --save_trajectory True

# We assume the Game Server running forever
INF = 4000000
SEED = 2999 

def parse_args():

    parser = argparse.ArgumentParser()

    # memo, hyper_index and server for create the serve
    parser.add_argument("--memo", type=str, default='ppo_pong')
    parser.add_argument("--server", type=str, default='pongdemo_adv')
    parser.add_argument("--mod", type=str, default="advtrain")

    # model_name (previous distinguish ppo2 and ppo1, now is ppo)
    parser.add_argument("--model_name", type=str, default="ppo1_oppomodel")
    parser.add_argument("--hyper_index", type=int, default=3)
    # seed value
    parser.add_argument("--x_method", type=str, default="grad")
    parser.add_argument("--mimic_model_path", type=str, default="../pretrain/saved/mimic_model.h5")
    parser.add_argument("--save_victim_traj", type=bool, default=False)
    parser.add_argument("--save_trajectory", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=0)

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    memo = args.memo
    server = args.server
    mode = args.mod

    model_name = args.model_name
    hyper_index = args.hyper_index

    x_method = args.x_method
    mimic_model_path = args.mimic_model_path
    save_victim_traj = args.save_victim_traj
    save_trajectory = args.save_trajectory
    seed = args.seed
    # create the gameserver, the same as enviroment
    game_server_id = server+"{0}".format(hyper_index)
    game = roboschool.gym_pong.PongSceneMultiplayer()
    gameserver = multiplayer.SharedMemoryServer(game, game_server_id, want_test_window=False, profix=str(hyper_index))

    player_0_args = "--memo={0} --server={1} " \
                "--mod={2} --model_name={3} --hyper_index={4} " \
                "--seed={5} --x_method={6} --mimic_model_path={7} --save_victim_traj={8} "\
                "--save_trajectory={9}".format(memo, game_server_id, mode,
                    model_name, hyper_index, seed, x_method, mimic_model_path, save_victim_traj, save_trajectory)

    # launch player0 (left player)
    player_0_args = player_0_args.split(" ")
    sys_cmd = [sys.executable, 'play_pong_player0.py']
    sys_cmd.extend(player_0_args)
    p0 = subprocess.Popen(sys_cmd)
    # launch player1 (right player)
    subprocess.Popen([sys.executable, 'play_pong_player1.py', game_server_id, "test"])
    
    import time
    start_time = time.time()
    try:
        gameserver.serve_forever(INF+1)
    except ValueError:
        print("===>",time.time()-start_time)
        print("End of training!")

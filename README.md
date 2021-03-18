# 1. Pong环境配置

## 1.1 安装必要的package

1. 安装openmpi: `sudo apt-get update && sudo apt-get install cmake libopenmpi-dev zlib1g-dev`
2. `sudo apt-get install libpcre3-dev libharfbuzz0b libgl1-mesa-glx`

## 1.2 安装Anaconda

* 下载：`wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh`
* 安装：`bash ./Anaconda3-2020.11-Linux-x86_64.sh`
* (should be changed)echo 'export PATH="$pathToAnaconda/anaconda3/bin:$PATH"' >> ~/.bashrc
* (optional) conda config --set auto_activate_base false

## 1.3 配置jupyter notebook

* 生成配置文件：jupyter notebook --generate-config
* 设置密码：jupyter notebook password
* 修改配置文件：vim ~/.jupyter/jupyter_notebook_config.py

```
# 添加(覆盖)下面内容
c.NotebookApp.ip = '*' # 开启所有的IP访问，即可使用远程访问
c.NotebookApp.open_browser = False # 关闭启动后的自动开启浏览器
c.NotebookApp.port = 8888  # 设置端口8888，也可用其他的，比如1080，8080等等
```

* 创建conda虚拟环境
    * 导出已有环境：在虚拟环境中执行conda env export > grid.yaml
    * 根据导出文件创建环境：conda env create -f grid.yaml
* 配置jupyter notebook kernel
    * `conda install ipykernel`
    * 激活虚拟环境，将环境写入notebook的kernel中, python -m ipykernel install --user --name pong --display-name pong
    
## 1.4 替换roboschool包里的相关文件

* run cd ~/anaconda3/envs/pong/lib/python3.6/site-packages/roboschool and copy the gym_pong.py, multiplayer.py and monitor.py files in the following folder https://drive.google.com/drive/folders/1A1U83hnu7S6kdVl6Q16ZW-hbNouigZF8?usp=sharing to the current folder roboschool.

* These files are stored in the 'replace' folder. `cp ./replace/* ~/anaconda3/envs/pong/lib/python3.6/site-packages/roboschool`

# 2. Pong实验流程
## 2.1 获得Agent Under Testing (AUT)

* 针对[roboschool agent_zoo](https://github.com/openai/roboschool/tree/master/agent_zoo)中提供的Pong_2017may1 agent，利用[stable baselines(v2.5.1)](https://stable-baselines.readthedocs.io/en/v2.5.0/)训练得到一个新的agent **AUT**， 并对其进行白盒场景下的分析测试。
    * `python play_pong_train.py --memo pong --server pongScene --mod ppotrain --model_name ppo1AUT --hyper_index 11 --x_method None --mimic_model_path None --oppo_name 2017may1 --save_oppo_traj ./pretrained/2017may1_against_ppo_traj.data --save_trajectory 1`
    * 训练过程中AUT的轨迹信息是否保存由`--save_trajectory 1`控制，若保存，则对应数据会和best_model存在同一个位置
    * 训练过程中对手的轨迹信息是否保存由`--save_oppo_traj ./pretrained/2017may1_against_ppo_traj.data`控制，若保存，则指定存储位置即可，默认为`''`不保存。
    * 同时运行多个程序时，用`--hyper_index 11`参数来区分，防止不同程序访问同一片共享内存。共享文件保存在`/tmp/`目录下，必要时可以清空上一次运行程序生成的文件`rm -rf multiplayer_p*`。

## 2.2 固定AUT，利用ppo1算法训练得到ppo1N (opponent agent) ，并保留训练阶段AUT的轨迹数据
    
* 修改`~/anaconda3/envs/pong/lib/python3.6/site-packages/stable_baselines/common/base_class.py` 中predict函数(line 463)：
    * 用`actions, _, states, _, last_pi = self.step(observation, state, mask, deterministic=deterministic)`替换line 472
    * 用`return clipped_actions, states, last_pi`替换line 485

* `python play_pong_train.py --memo pong --server pongScene --mod ppotrain --model_name ppo1N --hyper_index 11 --x_method None --mimic_model_path None --oppo_name AUT --save_oppo_traj ./Log/64-64-6-victim/model/AUT_against_ppoN.data --save_trajectory 0` # INF 小于3000000，即iteration小于3000即可，大概在2000+时，reward已经大于100

## 2.3 固定AUT，利用ppo1Adv算法训练得到ppo1Adv (opponent agent) ，并保留训练阶段AUT的轨迹数据
* 该算法默认的AUT是2017may2，理论上目前还不可以攻击其他AUT。但是在实际过程中，通过更换`play_pong_player1`中加载的模型，也是可以攻击成功的（可以理解为迁移性？）。
    * `python play_pong_train.py --memo pong --server pongScene --mod advtrain --model_name ppo1Adv --hyper_index 12 --x_method None --mimic_model_path None --oppo_name AUT --save_oppo_traj ./Log/64-64-6-victim/model/AUT_against_ppoAdv.data --save_trajectory 1 --save_victim_traj 1`
    * `play_pong_player0.py` line 176设置为白盒模式，即`hyper_weights = [0.0, -0.1, 0.0, 1, 0, 10, False, True, False]` 不加载pretrained mimic_model。
        * 但是这里的白盒不是真正的白盒，因为pposgd中的oppo_model是2017may2，与`play_pong_player1`中加载的模型不一致。


* 利用该算法时需要事先训练得到AUT的mimic model，从而进行后续的黑盒攻击。
    * 可以直接加载原作者github中提供的well-trained mimic model,存储位置：`./pretrained/o_mimic_model.h5`
    * 也可以利用另一个agent与AUT进行交互，得到trajectory信息，重新训练mimic model。


## 2.4 加载两个训练好的agent进行游戏
    
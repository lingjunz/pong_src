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

# 2. Pong实验设置
## 2.1 获得Agent Under Testing (AUT)

* 针对[roboschool agent_zoo](https://github.com/openai/roboschool/tree/master/agent_zoo)中提供的Pong_2017may1 agent，利用[stable baselines(v2.5.1)](https://stable-baselines.readthedocs.io/en/v2.5.0/)训练得到一个新的agent **AUT**， 并对其进行白盒场景下的分析测试。
    * `python play_pong_train.py --memo pong --server pongScene --mod ppotrain --model_name ppo1AUT --hyper_index 11 --x_method None --mimic_model_path None --oppo_name 2017may1 --save_victim_traj False --save_trajectory False`


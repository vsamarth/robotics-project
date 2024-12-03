from stable_baselines3 import DQN
from pogema import GridConfig, pogema_v0
from agent import 

env = pogema_v0(GridConfig(size=8, density=0.3, num_agents=1, max_episode_steps=30, integration="gymnasium"))
dqn_agent = DQN('MlpPolicy', env, verbose=1, tensorboard_log="./dqn_pogema_tensorboard/")

dqn_agent.learn(1000000, log_interval=1000, tb_log_name="baseline")
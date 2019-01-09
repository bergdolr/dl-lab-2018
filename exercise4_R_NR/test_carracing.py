from __future__ import print_function

import gym
from dqn.dqn_agent import DQNAgent
from train_carracing import run_episode
from dqn.networks import *
import numpy as np
import os
import json
from datetime import datetime
import argparse

np.random.seed(0)

if __name__ == "__main__":

    env = gym.make("CarRacing-v0").unwrapped
    
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="./models_carracing/dqn_agent.ckpt", type=str, nargs="?",
                        help="Path to the model")
    parser.add_argument("--episodes", default=15, type=int, nargs="?",
                        help="Number of test episodes")
    parser.add_argument("--history_length", default=0, type=int, nargs="?",
                        help="History length used for the given model")
    args = parser.parse_args()


    history_length =  args.history_length

    #TODO: Define networks and load agent
    # ....
    state_shape = [96, 96, 1+history_length]
    
    Q = CNN(state_shape, 5)
    Q_target = CNNTargetNetwork(state_shape, 5)
    
    agent = DQNAgent(Q,  Q_target,  5)
    
    agent.load(args.model)

    n_test_episodes = args.episodes

    episode_rewards = []
    for i in range(n_test_episodes):
        stats = run_episode(env, agent, deterministic=True, do_training=False, rendering=True)
        episode_rewards.append(stats.episode_reward)

    # save results in a dictionary and write them into a .json file
    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
 
    if not os.path.exists("./results"):
        os.mkdir("./results")  

    fname = "./results/carracing_results_dqn-%s.json" % datetime.now().strftime("%Y%m%d-%H%M%S")
    fh = open(fname, "w")
    json.dump(results, fh)
            
    env.close()
    print('... finished')


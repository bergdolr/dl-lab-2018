# export DISPLAY=:0 

import sys
sys.path.append("../") 

import numpy as np
import gym
from dqn.dqn_agent import DQNAgent
from dqn.networks import CNN, CNNTargetNetwork
from tensorboard_evaluation import *
import itertools as it
import argparse
from utils import EpisodeStats,  id_to_action,  rgb2gray

def run_episode(env, agent, deterministic, skip_frames=0,  do_training=True, rendering=False, max_timesteps=1000, history_length=0):
    """
    This methods runs one episode for a gym environment. 
    deterministic == True => agent executes only greedy actions according the Q function approximator (no random actions).
    do_training == True => train agent
    """

    stats = EpisodeStats()

    # Save history
    image_hist = []

    step = 0
    state = env.reset()

    # fix bug of corrupted states without rendering in gym environment
    env.viewer.window.dispatch_events() 

    # append image history to first state
    state = state_preprocessing(state)
    image_hist.extend([state] * (history_length + 1))
    state = np.array(image_hist).reshape(96, 96, history_length + 1)
    
    while True:

        # TODO: get action_id from agent
        # Hint: adapt the probabilities of the 5 actions for random sampling so that the agent explores properly. 
        action_id = agent.act(state=state, deterministic=deterministic,distr=[0.2, 0.05, 0.05, 0.65, 0.05])
        action = id_to_action(action_id)

        # Hint: frame skipping might help you to get better results.
        reward = 0
        for _ in range(skip_frames + 1):
            next_state, r, terminal, info = env.step(action)
            reward += r

            if rendering:
                env.render()

            if terminal: 
                 break

        next_state = state_preprocessing(next_state)
        image_hist.append(next_state)
        image_hist.pop(0)
        next_state = np.array(image_hist).transpose(1, 2, 0)

        if do_training:
            agent.train(state, action_id, next_state, reward, terminal)

        stats.step(reward, action_id)

        state = next_state
        
        if terminal or (step * (skip_frames + 1)) > max_timesteps : 
            break

        step += 1

    return stats


def train_online(env, agent, num_episodes, history_length=0, model_dir="./models_carracing", tensorboard_dir="./tensorboard",  eval_episodes=5, eval_interval=20,  eval_render=True,  quick_start=True,  skip_frames=0,  save_interval=50):
   
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train agent")
    tensorboard = Evaluation(os.path.join(tensorboard_dir, "train"), ["episode_reward", "straight", "left", "right", "accel", "brake", "eval_reward"])

    eval_reward = 0.0
    
    max_timesteps = 1000
    if quick_start:
        max_timesteps = 100
    for i in range(num_episodes):
        print("epsiode %d" % i)

        # Hint: you can keep the episodes short in the beginning by changing max_timesteps (otherwise the car will spend most of the time out of the track)
        if quick_start:
            if i == 100: max_timesteps = 500
            elif i == 500: max_timesteps = 1000
        #max_timesteps = min(1000,  50 +2*i)
        
        stats = run_episode(env, agent, max_timesteps=max_timesteps, deterministic=False, do_training=True, skip_frames=skip_frames,  history_length=history_length)

        # TODO: evaluate agent with deterministic actions from time to time
        # ...
        if eval_interval and (i % eval_interval == 0 or i >= (num_episodes - 1)):
            print("... starting greedy evaluation ...")
            eval_reward = 0.0
            for j in range(eval_episodes):
                eval_stats = run_episode(env, agent, deterministic=True, do_training=False,  rendering = eval_render,  history_length=history_length)
                eval_reward += eval_stats.episode_reward
            eval_reward /= eval_episodes
            print("... greedy evaluation reward: ",  eval_reward)

        tensorboard.write_episode_data(i, eval_dict={ "episode_reward" : stats.episode_reward, 
                                                                "straight" : stats.get_action_usage(0),
                                                                "left" : stats.get_action_usage(1),
                                                                "right" : stats.get_action_usage(2),
                                                                "accel" : stats.get_action_usage(3),
                                                                "brake" : stats.get_action_usage(4), 
                                                                "eval_reward": eval_reward})

        if i % save_interval == 0 or (i >= num_episodes - 1):
            agent.saver.save(agent.sess, os.path.join(model_dir, "dqn_agent.ckpt")) 

    tensorboard.close_session()

def state_preprocessing(state):
    return rgb2gray(state).reshape(96, 96) / 255.0

if __name__ == "__main__":

    env = gym.make('CarRacing-v0').unwrapped
    
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./models_carracing", type=str, nargs="?",
                        help="Path where the model will be stored")
    parser.add_argument("--input_model", default=None, type=str, nargs="?",
                        help="Path to an existing model to be trained further (if not given a new model will be built)")
    parser.add_argument("--episodes", default=1000, type=int, nargs="?",
                        help="Determines how many episodes the network will be trained")
    parser.add_argument("--history_length", default=0, type=int, nargs="?",
                        help="Length of the image history to be stored in each state")
    parser.add_argument("--skip_frames",  default=0,  type=int,  nargs="?", 
                        help="Number of frames to be skipped between frames recorded for training")
    parser.add_argument("--quick_start", default=True, type=bool, nargs="?", 
                        help="Use shorter episodes in the beginning")
    parser.add_argument("--eval_interval",  default=50, type=int,  nargs="?", 
                        help="Determines the number of episodes after which the agent will be evalauated periodically (0 means no evaluation during training)")
    parser.add_argument("--eval_episodes",  default=5,  type=int,  nargs="?", 
                        help="Determines over how man episodes the agent will be evaluated each time")
    parser.add_argument("--eval_render",  default=True,  type=bool,  nargs="?", 
                        help="Toggle rendering during evalluation")
    parser.add_argument("--save_interval",  default=100,  type=int,  nargs="?", 
                        help="Interval (in epsiodes) at which the agent is saved")
    args = parser.parse_args()
    
    # TODO: Define Q network, target network and DQN agent
    # ...
    
    history_length = args.history_length
    state_shape = [96, 96, 1+history_length]
    
    Q = CNN(state_shape, 5)
    Q_target = CNNTargetNetwork(state_shape, 5)
    
    agent = DQNAgent(Q,  Q_target,  5)
    
    if args.input_model:
        agent.load(args.input_model)
    
    train_online(env, agent, num_episodes=args.episodes, history_length=history_length, model_dir=args.model_dir, skip_frames= args.skip_frames,  quick_start=args.quick_start, eval_interval=args.eval_interval,  eval_episodes=args.eval_episodes, eval_render=args.eval_render,   save_interval=args.save_interval)


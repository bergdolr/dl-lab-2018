from __future__ import print_function

import pickle
import numpy as np
import os
import gzip
import random
import matplotlib.pyplot as plt

from model import Model
from utils import *
from tensorboard_evaluation import Evaluation

def read_data(datasets_dir="./data", frac = 0.1):
    """
    This method reads the states and actions recorded in drive_manually.py 
    and splits it into training/ validation set.
    """
    print("... read data")
    data_file = os.path.join(datasets_dir, 'data.pkl.gzip')
  
    f = gzip.open(data_file,'rb')
    data = pickle.load(f)

    # get images as features and actions as targets
    X = np.array(data["state"]).astype('float32')
    y = np.array(data["action"]).astype('float32')

    # split data into training and validation set
    n_samples = len(data["state"])
    X_train, y_train = X[:int((1-frac) * n_samples)], y[:int((1-frac) * n_samples)]
    X_valid, y_valid = X[int((1-frac) * n_samples):], y[int((1-frac) * n_samples):]
    return X_train, y_train, X_valid, y_valid


def preprocessing(X_train, y_train, X_valid, y_valid, history_length=1):

    # TODO: preprocess your data here.
    # 1. convert the images in X_train/X_valid to gray scale. If you use rgb2gray() from utils.py, the output shape (96, 96, 1)
    # 2. you can either train your model with continous actions (as you get them from read_data) using regression
    #    or you discretize the action space using action_to_id() from utils.py. If you discretize them, you'll maybe find one_hot() 
    #    useful and you may want to return X_train_unhot ... as well.
    n_train = X_train.shape[0]
    n_valid = X_valid.shape[0]
    dim_X = X_train.shape[1]
    
    X_train = rgb2gray(X_train)
    X_valid = rgb2gray(X_valid)
    
    y_train_ids = np.zeros(n_train)
    y_valid_ids = np.zeros(n_valid)
    for i in range(n_train):
        y_train_ids[i] = action_to_id(y_train[i])
    for i in range(n_valid):
        y_valid_ids[i] = action_to_id(y_valid[i])
    #print(y_train_ids.shape)
    y_train = one_hot(y_train_ids.astype(int))
    y_valid = one_hot(y_valid_ids.astype(int))
    # History:
    # At first you should only use the current image as input to your network to learn the next action. Then the input states
    # have shape (96, 96,1). Later, add a history of the last N images to your state so that a state has shape (96, 96, N).
    
    X_train = build_history(X_train,  n_train,  dim_X,  history_length)
    X_valid = build_history(X_valid,  n_valid,  dim_X,  history_length)
    
    #print(X_train.shape)
    #print(y_train.shape)
    
    return X_train, y_train, X_valid, y_valid

def build_history(X,  n,  dim,  history_length):
    X_hist = np.zeros((n,  dim,  dim,  history_length))
    for i in range(n):
        for j in range(history_length):
            X_hist[i, :, :, j] = X[max(0,  i-j),  :,  :]
    
    return X_hist

def train_model(X_train, y_train, X_valid, y_valid,  n_minibatches, batch_size, lr, model_dir="./models", tensorboard_dir="./tensorboard"):
    
    # create result and model folders
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)  
 
    print("... train model")


    # TODO: specify your neural network in model.py 
    agent = Model(X_train.shape[1:], y_train.shape[1],  lr)
    
    tensorboard_eval = Evaluation(tensorboard_dir)
    
    # TODO: implement the training
    # 
    # 1. write a method sample_minibatch and perform an update step
    # 2. compute training/ validation accuracy and loss for the batch and visualize them with tensorboard. You can watch the progress of
    #    your training in your web browser
    # 
    # training loop
    for i in range(n_minibatches):
        X_batch,  y_batch = sample_minibatch(X_train,  y_train,  batch_size)
        _,  loss = agent.session.run([agent.optimizer,  agent.loss],  
                                        feed_dict={agent.X : X_batch,  agent.y : y_batch})
        
        
        if i % 10 == 0:
            train_acc = agent.accuracy.eval({agent.X : X_batch,  agent.y : y_batch},  session=agent.session)
            val_acc = agent.accuracy.eval({agent.X : X_valid,  agent.y : y_valid},  session=agent.session)
            tensorboard_eval.write_episode_data(i,  {"loss" : agent.loss,  "train_acc" : train_acc,  "val_acc" : val_acc})
      
    # TODO: save your agent
    model_dir = agent.save(os.path.join(model_dir, "agent.ckpt"))
    print("Model saved in file: %s" % model_dir)


def sample_minibatch(X,  y,  batch_size):
    batch_indices = random.sample(range(len(X)),  batch_size)
    X_batch = X[batch_indices]
    y_batch = y[batch_indices]
    return X_batch,  y_batch

if __name__ == "__main__":

    # read data    
    X_train, y_train, X_valid, y_valid = read_data("./data")

    # preprocess data
    X_train, y_train, X_valid, y_valid = preprocessing(X_train, y_train, X_valid, y_valid, history_length=1)

    # train model (you can change the parameters!)
    train_model(X_train, y_train, X_valid, y_valid,  n_minibatches=1000, batch_size=64, lr=0.0001)
 

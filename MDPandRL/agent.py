'''
Group Members: Lalith Vennapusa, Oscar Ruenes
Date: 10/8/2025
Title: CS 4375 Assignment 3 Markov Decision Processes and Reinforcement Learning
Desc: Reinforcement learning algorithm for pole balancing using Q-learning. Implements a QLearningAgent to 
learn the optimal policy for balancing a pole on a cart. The environment simulates the physics of the pole and cart system.
'''
import math
import random

import numpy as np

FORWARD_ACCEL = 1
BACKWARD_ACCEL = 0


class QLearningAgent:
    def __init__(self, lr, gamma, track_length, epsilon=0, policy='greedy'):
        """
        A function for initializing your agent
        :param lr: learning rate
        :param gamma: discount factor
        :param track_length: how far the ends of the track are from the origin.
            e.g., while track_length is 2.4,
            the x-coordinate of the left end of the track is -2.4,
            the x-coordinate of the right end of the track is 2.4,
            and x-coordinate of the the cart is 0 initially.
        :param epsilon: epsilon for the mixed policy
        :param policy: can be 'greedy' or 'mixed'
        """
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.track_length = track_length
        self.policy = policy
        random.seed(11)
        np.random.seed(11)
        # you may add your code for initialization here, e.g., the Q-table
        pass

    def reset(self):
        """
        you may add code here to re-initialize your agent before each trial
        :return:
        """
        pass

    def get_action(self, x, x_dot, theta, theta_dot):
        """
        main.py calls this method to get an action from your agent
        :param x: the position of the cart
        :param x_dot: the velocity of the cart
        :param theta: the angle between the cart and the pole
        :param theta_dot: the angular velocity of the pole
        :return:
        """
        if self.policy == 'mixed' and random.random() < self.epsilon:
            action = random.sample([FORWARD_ACCEL, BACKWARD_ACCEL], 1)[0]
        else:
            # fill your code here to get an action from your agent
            action = None
        return action

    def update_Q(self, prev_state, prev_action, cur_state, reward):
        """
        main.py calls this method so that you can update your Q-table
        :param prev_state: previous state, a tuple of (x, x_dot, theta, theta_dot)
        :param prev_action: previous action, FORWARD_ACCEL or BACKWARD_ACCEL
        :param cur_state: current state, a tuple of (x, x_dot, theta, theta_dot)
        :param reward: reward, 0.0 or -1.0
        e.g., if we have S_i ---(action a, reward)---> S_j, then
            prev_state is S_i,
            prev_action is a,
            cur_state is S_j,
            rewards is reward.
        :return:
        """
        # fill your code here to update your Q-table
        raise NotImplementedError

    # you may add more methods here for your needs. E.g., methods for discretizing the variables.


if __name__ == '__main__':
    pass
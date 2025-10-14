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
X_STEP = 2
X_DOT_STEP = 0.1
THETA_STEP = 0.1
THETA_DOT_STEP = 0.25


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
        #we will have q_table such that an entry in the queue table is (state_tuple, action) -> real Q value
        self.q_table = dict()
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
            state = self.discretize_state(x, x_dot, theta, theta_dot)
            action = FORWARD_ACCEL if self.get_q(state, FORWARD_ACCEL) >= self.get_q(state, BACKWARD_ACCEL) else BACKWARD_ACCEL
            
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
        prev_state_disc = self.discretize_state(*prev_state)
        cur_state_disc = self.discretize_state(*cur_state)
        old_q = self.get_q(prev_state_disc, prev_action)
        new_q = (1 - self.lr) * old_q + self.lr * (reward + self.gamma * max(self.get_q(cur_state_disc, a) for a in [FORWARD_ACCEL, BACKWARD_ACCEL]))
        self.q_table[(prev_state_disc, prev_action)] = new_q
        pass

    # you may add more methods here for your needs. E.g., methods for discretizing the variables.
    def discretize_state(self, x, x_dot, theta, theta_dot):
        return (round(x / X_STEP) * X_STEP,
            round(x_dot / X_DOT_STEP) * X_DOT_STEP,
            round(theta / THETA_STEP) * THETA_STEP,
            round(theta_dot / THETA_DOT_STEP) * THETA_DOT_STEP)
    
    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)




if __name__ == '__main__':
    pass
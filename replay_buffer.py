import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import os


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity
        self.device = device

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((self.capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.last_save = 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size, discount):
        idxs = np.random.randint(0,
                                 self.capacity if self.full else self.idx,
                                 size=batch_size)

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        next_obses = torch.as_tensor(self.next_obses[idxs],
                                     device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        discounts = np.ones((idxs.shape[0], 1), dtype=np.float32) * discount
        discounts = torch.as_tensor(discounts, device=self.device)

        return obses, actions, rewards, next_obses, discounts

class ReplayBufferObsbank(object):
    def __init__(self, obs_shape, action_shape, capacity, horizon, device):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.capacity = capacity
        self.horizon = horizon
        self.device = device

        self.samplable = np.zeros((capacity, 1), dtype=bool) 
        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        # self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        # self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        self.last_save = 0
        self.start_idx = 0
        self.idx_arange = np.ones([self.capacity], dtype=np.uint32)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, done):  # reward, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        # np.copyto(self.rewards[self.idx], reward)
        # np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        if done:
            samplable_start_idx = max(self.start_idx, self.idx-self.horizon+1)
            np.copyto(self.samplable[samplable_start_idx:self.idx+1], False)
        else:
            np.copyto(self.samplable[self.idx], True)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
        self.start_idx = self.idx

    def sample(self, batch_size):  #, discount):
        # idxs = np.random.randint(0,
        #                          self.capacity if self.full else self.idx,
        #                          size=batch_size)
        batch_arange = np.tile(np.arange(self.horizon), [batch_size, 1])
        idxs = np.random.choice(self.idx_arange[self.samplable[:, 0]], [batch_size, 1])
        idxs_batch = idxs + batch_arange

        obses = torch.as_tensor(self.obses[idxs[:, 0]], device=self.device).float()
        next_obses_h = torch.as_tensor(self.obses[idxs_batch+1],
                                     device=self.device).float()
        actions_h = torch.as_tensor(self.actions[idxs_batch], device=self.device)
        # rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        # discounts = np.ones((idxs.shape[0], 1), dtype=np.float32) * discount
        # discounts = torch.as_tensor(discounts, device=self.device)

        return obses, actions_h, next_obses_h  #, rewards, next_obses, discounts
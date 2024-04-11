"""Data formating"""

# import torch
import random

import matplotlib.pyplot as plt
import numpy as np  # for the math


class dataLoader:
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

        self.data_size = inputs.shape[0]
        self.cov_dim = inputs.shape[1]
        self.input_shape = inputs.shape
        self.output_shape = outputs.shape

        self.IDs = list(range(inputs.shape[0]))
        self.train_ids = self.IDs

    def set_train_ids(self, K=None):
        if K == None:
            self.train_ids = self.IDs
        else:
            self.train_ids = random.sample(self.IDs, K)

    def get_input(self, id=None):
        if id != None:
            return np.delete(self.inputs[self.train_ids, :], id, 0)
        else:
            return self.inputs[self.train_ids, :]

    def get_output(self, id=None):
        if id != None:
            return np.delete(self.outputs[self.train_ids, :], id, 0)
        else:
            return self.outputs[self.train_ids, :]

    # def get_input_tensor(self, id=None):
    #     if id != None:
    #         return torch.Tensor(np.delete(self.inputs[self.train_ids, :], id, 0))
    #     else:
    #         return torch.Tensor(self.inputs[self.train_ids, :])

    # def get_output_tensor(self, id=None):
    #     if id != None:
    #         return torch.Tensor(np.delete(self.outputs[self.train_ids, :], id, 0))
    #     else:
    #         return torch.Tensor(self.outputs[self.train_ids, :])

    def get_size(self):
        return self.data_size

    def get_cur_size(self):
        return len(self.train_ids)

    def get_cov_dim(self):
        return self.cov_dim

    def add2_inputs(self, newX):
        self.inputs = np.concatenate((self.inputs, newX))
        self.train_ids.append(self.data_size)

    def add2_outputs(self, newY):
        self.outputs = np.concatenate((self.outputs, newY))

    def rm_lst_inputs(self):
        self.inputs = np.delete(self.inputs, -1, 0)
        self.train_ids = self.train_ids[:-1]

    def rm_lst_outputs(self):
        self.outputs = np.delete(self.outputs, -1, 0)

    def plot(self):
        plt.scatter(
            self.inputs[self.train_ids, :], self.outputs[self.train_ids, :], marker="+"
        )

import json
import os
import os.path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# import preprocess_lstm
import re
import os
import sys
import pickle as cPickle

import time
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
# import data
import pickle
import eval
import training_dataset_class
import val_dataset_class


def batch_accuracy(predicted, true):
    _, predicted_index = predicted.max(dim=1, keepdim=True)
    # print('predicted size= ', predicted.size())
    # print('true size= ', true.size())
    predicted_index = torch.squeeze(predicted_index.view(true.size(0), -1))
    # predicted_index = predicted_index.view(-1,true.size(1))
    # print('predicted_index size= ', predicted_index.size())
    # acc = min(torch.eq(true, predicted_index).sum().item()* 0.3, 3)
    acc = 100 * torch.eq(true, predicted_index).sum().item() / len(true)
    # print('answer = ',answer)
    # print('pred_idx= ',predicted_index)
    # agreeing = true.gather(dim=1, index=predicted_index)
    return acc
    # return (agreeing * 0.3).clamp(max=1)

def train():
    os.system('python /home/student/hw2_dl/train.py')
    return
def evaluate_hw2():
    os.system('python /home/student/hw2_dl/eval.py')
    return

if __name__ == '__main__':
    start = time.time()
    evaluate_hw2()




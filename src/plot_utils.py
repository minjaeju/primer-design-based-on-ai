from __future__ import unicode_literals, print_function, division
import unicodedata
import string
import re
import random

import time
import math

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as dates
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

    
def show_plot(points, plot_every, fold, save_path='./result', file_name='train', save_as_img=True):
    plt.figure()
    fig, ax = plt.subplots()

    plt.title('%s %d fold loss' % (file_name, fold))
    x = list(range(1, len(points)*plot_every+1, plot_every))
    plt.plot(x, points)

    if save_as_img:
        plt.savefig(os.path.join(save_path, '%s_fold%d.png' % (file_name, fold)))

        
def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

    
def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))

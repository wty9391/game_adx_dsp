import sys
import pickle
import numpy as np
import random

import adx


# ./result/2997

if len(sys.argv) < 2:
    print('Usage: .py result_root_path')
    exit(-1)


x_test = pickle.load(open(sys.argv[1] + '/x_test', 'rb'))
y_test = pickle.load(open(sys.argv[1] + '/y_test', 'rb'))
z_test = pickle.load(open(sys.argv[1] + '/z_test', 'rb'))

adx_instance = pickle.load(open(sys.argv[1] + '/adx', 'rb'))
adx_instance.play(sys.argv[1].split("/")[-1], x_test, y_test, z_test,
                  dsp_total_budget=z_test.sum()/1, game_interval=2e4)


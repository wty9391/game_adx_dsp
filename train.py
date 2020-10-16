import sys
import pickle
import numpy as np
import random

import adx

# ./result/2997

if len(sys.argv) < 2:
    print('Usage: .py result_root_path')
    exit(-1)

dsp_number = 5

x_train = pickle.load(open(sys.argv[1] + '/x_train', 'rb'))
y_train = pickle.load(open(sys.argv[1] + '/y_train', 'rb'))
z_train = pickle.load(open(sys.argv[1] + '/z_train', 'rb'))

(train_size, _) = np.shape(x_train)


index = np.arange(train_size)
random.shuffle(index)
indices = np.array_split(index, dsp_number)

adx_instance = adx.ADX(dsp_number=dsp_number)
adx_instance.train_dsp(x_train, y_train, z_train, indices)
adx_instance.train_adx(x_train, y_train, z_train)

pickle.dump(adx_instance, open(sys.argv[1] + '/adx', 'wb'))


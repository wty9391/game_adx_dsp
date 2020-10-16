import sys
import pickle
import numpy as np
from scipy.sparse import csr_matrix, vstack

from util import encoder

# /home/wty/datasets/make-ipinyou-data/2997/train.log.txt /home/wty/datasets/make-ipinyou-data/2997/test.log.txt /home/wty/datasets/make-ipinyou-data/2997/featindex.txt ./result/2997


if len(sys.argv) < 5:
    print('Usage: .py trian_log_path test_log_path feat_path result_root_path')
    exit(-1)

read_batch_size = 1e6

f_train_log = open(sys.argv[1], 'r', encoding="utf-8")
f_test_log = open(sys.argv[2], 'r', encoding="utf-8")

# init name_col
name_col = {}
s = f_train_log.readline().split('\t')
for i in range(0, len(s)):
    name_col[s[i].strip()] = i

ipinyou = encoder.Encoder_ipinyou(sys.argv[3], name_col)
X_train_raw = []
X_train = csr_matrix((0, len(ipinyou.feat)), dtype=np.int8)
Y_train = np.zeros((0, 1), dtype=np.int8)
B_train = np.zeros((0, 1), dtype=np.int16)
Z_train = np.zeros((0, 1), dtype=np.int16)
X_test_raw = []
X_test = csr_matrix((0, len(ipinyou.feat)), dtype=np.int8)
Y_test = np.zeros((0, 1), dtype=np.int8)
B_test = np.zeros((0, 1), dtype=np.int16)
Z_test = np.zeros((0, 1), dtype=np.int16)

count = 0
f_train_log.seek(0)
f_train_log.readline()  # first line is header
for line in f_train_log:
    X_train_raw.append(line)
    count += 1
    if count % read_batch_size == 0:
        X_train = vstack((X_train, ipinyou.encode(X_train_raw)))
        Y_train = np.vstack((Y_train, ipinyou.get_col(X_train_raw, "click")))
        Z_train = np.vstack((Z_train, ipinyou.get_col(X_train_raw, "payprice")))
        X_train_raw = []
if X_train_raw:
    X_train = vstack((X_train, ipinyou.encode(X_train_raw)))
    Y_train = np.vstack((Y_train, ipinyou.get_col(X_train_raw, "click")))
    Z_train = np.vstack((Z_train, ipinyou.get_col(X_train_raw, "payprice")))
    X_train_raw = []

count = 0
f_test_log.seek(0)
f_test_log.readline()  # first line is header
for line in f_test_log:
    X_test_raw.append(line)
    count += 1
    if count % read_batch_size == 0:
        X_test = vstack((X_test, ipinyou.encode(X_test_raw)))
        Y_test = np.vstack((Y_test, ipinyou.get_col(X_test_raw, "click")))
        Z_test = np.vstack((Z_test, ipinyou.get_col(X_test_raw, "payprice")))
        X_test_raw = []
if X_test_raw:
    X_test = vstack((X_test, ipinyou.encode(X_test_raw)))
    Y_test = np.vstack((Y_test, ipinyou.get_col(X_test_raw, "click")))
    Z_test = np.vstack((Z_test, ipinyou.get_col(X_test_raw, "payprice")))
    X_test_raw = []

(train_size, x_dimension) = np.shape(X_train)
(test_size, _) = np.shape(X_test)


pickle.dump(X_train, open(sys.argv[4] + '/x_train', 'wb'))
pickle.dump(Y_train, open(sys.argv[4]+'/y_train', 'wb'))
pickle.dump(Z_train, open(sys.argv[4] + '/z_train', 'wb'))

pickle.dump(X_test, open(sys.argv[4] + '/x_test', 'wb'))
pickle.dump(Y_train, open(sys.argv[4]+'/y_test', 'wb'))
pickle.dump(Z_train, open(sys.argv[4] + '/z_test', 'wb'))

f_train_log.close()
f_test_log.close()

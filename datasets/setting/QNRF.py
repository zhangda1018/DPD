from easydict import EasyDict as edict


# init
__C_QNRF = edict()

cfg_data = __C_QNRF

__C_QNRF.TRAIN_SIZE = (384,384)
__C_QNRF.DATA_PATH = '/data/gjy/Datas/QNRF'
__C_QNRF.TRAIN_LST = 'train.txt'
__C_QNRF.VAL_LST =  'test.txt'
__C_QNRF.VAL4EVAL = 'test_gt_loc.txt'

__C_QNRF.MEAN_STD = ([0.413525998592, 0.378520160913, 0.371616870165], [0.284849464893, 0.277046442032, 0.281509846449])

__C_QNRF.LABEL_FACTOR = 1
__C_QNRF.LOG_PARA = 1.

__C_QNRF.RESUME_MODEL = ''#model path
__C_QNRF.TRAIN_BATCH_SIZE = 8 #imgs

__C_QNRF.VAL_BATCH_SIZE = 1 # must be 1
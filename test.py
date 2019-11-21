import argparse
from functools import partial
import json

import numpy as np
import os
import tensorflow as tf
import utils as utf

import data
import models
import imageio

# ==============================================================================
# =                                    param                                   =
# ==============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--experiment_name', dest='experiment_name', help='experiment_name')
parser.add_argument('--test_int', dest='test_int', type=float, default=1.0, help='test_int')
args_ = parser.parse_args()
# with open('./output/%s/setting.txt' % args_.experiment_name) as f:
#     args = json.load(f)
with open('output/128_shortcut1_inject1_none_40/setting.txt') as f:
    args = json.load(f)

# model
atts = args['atts']
n_att = len(atts)
img_size = args['img_size']
shortcut_layers = args['shortcut_layers']
inject_layers = args['inject_layers']
enc_dim = args['enc_dim']
dec_dim = args['dec_dim']
dis_dim = args['dis_dim']
dis_fc_dim = args['dis_fc_dim']
enc_layers = args['enc_layers']
dec_layers = args['dec_layers']
dis_layers = args['dis_layers']
# testing
thres_int = args['thres_int']
# test_int = args_.test_int
test_int = 1.0
# others
use_cropped_img = args['use_cropped_img']
# experiment_name = args_.experiment_name
experiment_name = '128_shortcut1_inject1_none_40'

# ==============================================================================
# =                                   graphs                                   =
# ==============================================================================

# data
sess = utf.session()
te_data = data.Celeba('./data', atts, img_size, 1, part='test', sess=sess, crop=not use_cropped_img)

# models
Genc = partial(models.Genc, dim=enc_dim, n_layers=enc_layers)
Gdec = partial(models.Gdec, dim=dec_dim, n_layers=dec_layers, shortcut_layers=shortcut_layers, inject_layers=inject_layers)

# inputs
xa_sample = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3])
_b_sample = tf.placeholder(tf.float32, shape=[None, n_att])

# sample
x_sample = Gdec(Genc(xa_sample, is_training=False), _b_sample, is_training=False)


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# initialization
ckpt_dir = './output/%s/checkpoints' % experiment_name
utf.load_checkpoint(ckpt_dir, sess)

# sample
for idx, batch in enumerate(te_data):
    xa_sample_ipt = batch[0]
    a_sample_ipt = batch[1]
    b_sample_ipt_list = [a_sample_ipt]  # the first is for reconstruction
    for i in range(len(atts)):
        tmp = np.array(a_sample_ipt, copy=True)
        tmp[:, i] = 1 - tmp[:, i]   # inverse attribute
        tmp = data.Celeba.check_attribute_conflict(tmp, atts[i], atts)
        b_sample_ipt_list.append(tmp)

    x_sample_opt_list = [xa_sample_ipt, np.full((1, img_size, img_size // 10, 3), -1.0)]
    for i, b_sample_ipt in enumerate(b_sample_ipt_list):
        _b_sample_ipt = (b_sample_ipt * 2 - 1) * thres_int
        if i > 0:   # i == 0 is for reconstruction
            _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * test_int / thres_int
        x_sample_opt_list.append(sess.run(x_sample, feed_dict={xa_sample: xa_sample_ipt, _b_sample: _b_sample_ipt}))
    sample = np.concatenate(x_sample_opt_list, 2)

    save_dir = './output/%s/sample_testing' % experiment_name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    imageio.imwrite('%s/%d.png' % (save_dir, idx + 13000), sample.squeeze(0))

    print('%d.png done!' % (idx + 182638))

sess.close()
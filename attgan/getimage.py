from functools import partial

import numpy as np
import os
import tensorflow as tf
import attgan.utils as utf

import attgan.data as data
import attgan.models as models
import imageio

class _attgan:

    def __init__(self):

        self.atts = ["Bald",
        "Bangs",
        "Black_Hair",
        "Blond_Hair",
        "Brown_Hair",
        "Bushy_Eyebrows",
        "Eyeglasses",
        "Male",
        "Mouth_Slightly_Open",
        "Mustache",
        "No_Beard",
        "Pale_Skin",
        "Young"]
        self.n_att = len(self.atts)
        self.img_size = 384
        self.shortcut_layers = 1
        self.inject_layers = 1
        self.enc_dim = 48
        self.dec_dim = 48
        self.dis_dim = 48
        self.dis_fc_dim = 512
        self.enc_layers = 5
        self.dec_layers = 5
        self.dis_layers = 5
        self.thres_int = 0.5
        self.test_int = 1.0
        self.use_cropped_img = True
        self.experiment_name = '384_shortcut1_inject1_none_hd'


        self.sess = utf.session()
        self.te_data = data.Celeba('./data', self.img_size, 1, sess=self.sess, crop=not self.use_cropped_img)

        Genc = partial(models.Genc, dim=self.enc_dim, n_layers=self.enc_layers)
        Gdec = partial(models.Gdec, dim=self.dec_dim, n_layers=self.dec_layers,
                       shortcut_layers=self.shortcut_layers, inject_layers=self.inject_layers)

        self.xa_sample = tf.placeholder(tf.float32, shape=[None, self.img_size, self.img_size, 3])
        self._b_sample = tf.placeholder(tf.float32, shape=[None, self.n_att])

        self.x_sample = Gdec(Genc(self.xa_sample, is_training=False), self._b_sample, is_training=False)

        ckpt_dir = './model/%s/checkpoints' % self.experiment_name
        utf.load_checkpoint(ckpt_dir, self.sess)

    def getimage(self, att):

        sample = None

        for idx, batch in enumerate(self.te_data):
            xa_sample_ipt = batch[0]
            a_sample_ipt = batch[1]
            b_sample_ipt_list = [a_sample_ipt]
            for i in range(len(self.atts)):
                tmp = np.array(a_sample_ipt, copy=True)
                tmp[:, i] = 1 - tmp[:, i]
                tmp = data.Celeba.check_attribute_conflict(tmp, self.atts[i], self.atts)
                b_sample_ipt_list.append(tmp)

            x_sample_opt_list = [xa_sample_ipt, np.full((1, self.img_size, self.img_size // 10, 3), -1.0)]
            for i, b_sample_ipt in enumerate(b_sample_ipt_list):
                _b_sample_ipt = (b_sample_ipt * 2 - 1) * self.thres_int
                if i > 0:
                    _b_sample_ipt[..., i - 1] = _b_sample_ipt[..., i - 1] * self.test_int / self.thres_int
                x_sample_opt_list.append(self.sess.run(self.x_sample,
                                                       feed_dict={self.xa_sample: xa_sample_ipt,
                                                                  self._b_sample: _b_sample_ipt}))
            x_sample_opt_list = x_sample_opt_list[3:]
            sample = np.concatenate(x_sample_opt_list[self.atts.index(att)], 2)

            # save_dir = '../output'
            # if not os.path.isdir(save_dir):
            #     os.mkdir(save_dir)
            # imageio.imwrite('%s/%d.png' % (save_dir, idx + 1), sample)

            # print('%d.png done!' % (idx + 1))

        self.sess.close()

        return sample

import time
start = time.time()
att = _attgan()
att.getimage("Blond_Hair")
print(start - time.time())
import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from attgan.utils import session

from slimcnn.getattr import slimcnn

def batch_dataset(dataset, batch_size, prefetch_batch=2, drop_remainder=True, filter=None,
                  map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):
    if filter:
        dataset = dataset.filter(filter)

    if map_func:
        dataset = dataset.map(map_func, num_parallel_calls=num_threads)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    if drop_remainder:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        dataset = dataset.batch(batch_size)

    dataset = dataset.repeat(repeat).prefetch(prefetch_batch)

    return dataset

def disk_image_batch_dataset(img_paths, batch_size, labels=None, prefetch_batch=2, drop_remainder=True, filter=None,
                             map_func=None, num_threads=16, shuffle=True, buffer_size=4096, repeat=-1):

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))

    def parse_func(path, *label):
        img = tf.read_file(path)
        img = tf.image.decode_png(img, 3)
        return (img,) + label

    def map_func_(*args):
        return map_func(*parse_func(*args))

    dataset = batch_dataset(dataset, batch_size, prefetch_batch, drop_remainder, filter,
                            map_func_, num_threads, shuffle, buffer_size, repeat)

    return dataset

class Dataset(object):

    def __init__(self):
        self._dataset = None
        self._iterator = None
        self._batch_op = None
        self._sess = None

        self._is_eager = tf.executing_eagerly()
        self._eager_iterator = None

    def __del__(self):
        if self._sess:
            self._sess.close()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            b = self.get_next()
        except:
            raise StopIteration
        else:
            return b

    next = __next__

    def get_next(self):
        if self._is_eager:
            return self._eager_iterator.get_next()
        else:
            return self._sess.run(self._batch_op)

    def reset(self, feed_dict={}):
        if self._is_eager:
            self._eager_iterator = tfe.Iterator(self._dataset)
        else:
            self._sess.run(self._iterator.initializer, feed_dict=feed_dict)

    def _bulid(self, dataset, sess=None):
        self._dataset = dataset

        if self._is_eager:
            self._eager_iterator = tfe.Iterator(dataset)
        else:
            self._iterator = dataset.make_initializable_iterator()
            self._batch_op = self._iterator.get_next()
            if sess:
                self._sess = sess
            else:
                self._sess = session()

        try:
            self.reset()
        except:
            pass

    @property
    def dataset(self):
        return self._dataset

    @property
    def iterator(self):
        return self._iterator

    @property
    def batch_op(self):
        return self._batch_op


class Celeba(Dataset):

    att_dict = {'5_o_Clock_Shadow': 0, 'Arched_Eyebrows': 1, 'Attractive': 2,
                'Bags_Under_Eyes': 3, 'Bald': 4, 'Bangs': 5, 'Big_Lips': 6,
                'Big_Nose': 7, 'Black_Hair': 8, 'Blond_Hair': 9, 'Blurry': 10,
                'Brown_Hair': 11, 'Bushy_Eyebrows': 12, 'Chubby': 13,
                'Double_Chin': 14, 'Eyeglasses': 15, 'Goatee': 16,
                'Gray_Hair': 17, 'Heavy_Makeup': 18, 'High_Cheekbones': 19,
                'Male': 20, 'Mouth_Slightly_Open': 21, 'Mustache': 22,
                'Narrow_Eyes': 23, 'No_Beard': 24, 'Oval_Face': 25,
                'Pale_Skin': 26, 'Pointy_Nose': 27, 'Receding_Hairline': 28,
                'Rosy_Cheeks': 29, 'Sideburns': 30, 'Smiling': 31,
                'Straight_Hair': 32, 'Wavy_Hair': 33, 'Wearing_Earrings': 34,
                'Wearing_Hat': 35, 'Wearing_Lipstick': 36,
                'Wearing_Necklace': 37, 'Wearing_Necktie': 38, 'Young': 39}

    def __init__(self, data_dir, img_resize, batch_size, prefetch_batch=2,
                 num_threads=16, buffer_size=4096, sess=None, crop=True):
        super(Celeba, self).__init__()

        slimnet = slimcnn()

        img_dir = os.path.join(data_dir, 'img_align_celeba_png')

        names = ["dummy.png", "result.png"]
        img_paths = [os.path.join(img_dir, name.replace('.jpg', '.png')) for name in names]
        labels = list(map(slimnet.get_attr, img_paths))

        offset_h = 40
        offset_w = 15
        img_size = 148

        def _map_func(img, label):
            if crop:
                img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, img_size, img_size)
            img = tf.image.resize_images(img, [img_resize, img_resize], tf.image.ResizeMethod.BICUBIC)
            img = tf.clip_by_value(img, 0, 255) / 127.5 - 1
            label = (label + 1) // 2
            return img, label

        drop_remainder = False
        shuffle = False
        repeat = 1

        dataset = disk_image_batch_dataset(img_paths=img_paths,
                                           labels=labels,
                                           batch_size=batch_size,
                                           prefetch_batch=prefetch_batch,
                                           drop_remainder=drop_remainder,
                                           map_func=_map_func,
                                           num_threads=num_threads,
                                           shuffle=shuffle,
                                           buffer_size=buffer_size,
                                           repeat=repeat)
        self._bulid(dataset, sess)

        self._img_num = len(img_paths)

    def __len__(self):
        return self._img_num

    @staticmethod
    def check_attribute_conflict(att_batch, att_name, att_names):
        def _set(att, value, att_name):
            if att_name in att_names:
                att[att_names.index(att_name)] = value

        att_id = att_names.index(att_name)

        for att in att_batch:
            if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] == 1:
                _set(att, 0, 'Bangs')
            elif att_name == 'Bangs' and att[att_id] == 1:
                _set(att, 0, 'Bald')
                _set(att, 0, 'Receding_Hairline')
            elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] == 1:
                for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                    if n != att_name:
                        _set(att, 0, n)
            elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] == 1:
                for n in ['Straight_Hair', 'Wavy_Hair']:
                    if n != att_name:
                        _set(att, 0, n)

        return att_batch

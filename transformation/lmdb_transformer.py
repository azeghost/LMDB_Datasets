import os
import pickle
import numpy as np

from tqdm import tqdm
import lmdb
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from transformation.file_image_generator import create_image_lists
from transformation.file_utils import get_file_path, create_if_not_exist
from transformation.wrappers import DatasetWrapper


class LmdbTransformer:
    def __init__(self, validation_pct, valid_image_formats, image_dir=None, data_format=None, scalar=255.0):
        if image_dir is not None:
            self.image_lists = create_image_lists(image_dir, validation_pct, valid_image_formats)
        self.scaler = scalar
        if data_format is None:
            self.data_format = K.image_data_format()
        else:
            self.data_format = data_format

    def store_single_lmdb(self, filename, img, index, labels_dict, num_images):
        """ Stores a wrapper to LMDB.
        """
        map_size = num_images * img.nbytes * 10
        env = lmdb.open(filename, map_size=map_size)

        # Same as before — but let's write all the images in a single transaction
        with env.begin(write=True) as txn:
            # All key-value pairs need to be Strings
            value = DatasetWrapper(img, labels_dict)
            key = f"{index:08}"
            txn.put(key.encode("ascii"), pickle.dumps(value))

        env.close()

    def transform_store_from_numpy(self, images, labels, lmdb_dir='.data/', category='training',
                                   target_size=None, color_mode='rgb'):
        create_if_not_exist(lmdb_dir)
        num_images = labels.shape[0]
        lmdb_name = lmdb_dir + os.sep + '_{}'.format(category)
        index = 0
        print('Storing ' + str(num_images) + lmdb_dir + ' _{}'.format(category))
        for idx, (image, label) in tqdm(enumerate(zip(images, labels)), total=num_images):
            img = np.float32(image) / self.scaler

            self.store_single_lmdb(index=index, filename=lmdb_name, img=img, labels_dict={'label1': str(label)},
                                   num_images=num_images)
            index = index + 1

    def transform_store(self, image_dir, labels_fn,
                        lmdb_dir='.data/', category='training', target_size=None,
                        color_mode='rgb'):
        create_if_not_exist(lmdb_dir)

        classes = list(self.image_lists.keys())
        num_class = len(classes)
        class2id = dict(zip(classes, range(len(classes))))
        id2class = dict((v, k) for k, v in class2id.items())

        for label_name in classes:
            num_images = len(self.image_lists[label_name][category])
            print('Storing ' + str(num_images) + lmdb_dir + os.sep + '_{}'.format(category))
            for index, _ in enumerate(self.image_lists[label_name][category]):
                img_path = get_file_path(self.image_lists,
                                         label_name,
                                         index,
                                         image_dir,
                                         category)

                img = img_to_array(
                    load_img(
                        img_path,
                        grayscale=color_mode == 'grayscale',
                        target_size=target_size
                    ), data_format=self.data_format
                ) / self.scaler
                label = labels_fn(img_path)
                name = lmdb_dir + os.sep + '_{}'.format(category)
                self.store_single_lmdb(index=index, filename=name, img=img, labels_dict=label, num_images=num_images)

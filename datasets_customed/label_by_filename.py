import os
import numpy as np


def get_label_by_filename(img_path):
    name, _ = os.path.splitext(img_path)
    vid_img_arr = name.split(sep=os.sep)[-1:]
    return {'label': np.array(vid_img_arr)}
    # return [np.array(vid_img_arr)]

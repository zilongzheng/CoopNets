from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np
from PIL import Image
import scipy.misc

from six.moves import xrange

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

class DataSet(object):
    def __init__(self, data_path, image_size=128):
        self.root_dir = data_path
        self.imgList = [f for f in os.listdir(data_path) if any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS)]
        self.imgList.sort()
        self.image_size = image_size
        self.images = np.zeros((len(self.imgList), image_size, image_size, 3)).astype(float)
        print('Loading dataset: {}'.format(data_path))
        for i in xrange(len(self.imgList)):
            image = Image.open(os.path.join(self.root_dir, self.imgList[i])).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = np.array(image).astype(float)
            max_val = image.max()
            min_val = image.min()
            image = (image - min_val) / (max_val - min_val) * 2 - 1
            self.images[i] = image
        print('Data loaded, shape: {}'.format(self.images.shape))

    def data(self):
        return self.images

    def mean(self):
        return np.mean(self.images, axis=(0, 1, 2, 3))

    def to_range(self, low_bound, up_bound):
        min_val = self.images.min()
        max_val = self.images.max()
        return low_bound + (self.images - min_val) / (max_val - min_val) * (up_bound - low_bound)

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.imgList)

def mkdir(path, max_depth=3):
    parent, child = os.path.split(path)
    if not os.path.exists(parent) and max_depth > 1:
        mkdir(parent, max_depth-1)

    if not os.path.exists(path):
        os.mkdir(path)

def cell2img(cell_image, image_size=100, margin_syn=2):
    num_cols = cell_image.shape[1] // image_size
    num_rows = cell_image.shape[0] // image_size
    images = np.zeros((num_cols * num_rows, image_size, image_size, 3))
    for ir in range(num_rows):
        for ic in range(num_cols):
            temp = cell_image[ir*(image_size+margin_syn):image_size + ir*(image_size+margin_syn),
                   ic*(image_size+margin_syn):image_size + ic*(image_size+margin_syn),:]
            images[ir*num_cols+ic] = temp
    return images

def clip_by_value(input_, min=0, max=1):
    return np.minimum(max, np.maximum(min, input_))

def img2cell(images, row_num=10, col_num=10, margin_syn=2):
    [num_images, image_size] = images.shape[0:2]
    num_cells = int(math.ceil(num_images / (col_num * row_num)))
    cell_image = np.zeros((num_cells, row_num * image_size + (row_num-1)*margin_syn,
                           col_num * image_size + (col_num-1)*margin_syn, 3))
    for i in range(num_images):
        cell_id = int(math.floor(i / (col_num * row_num)))
        idx = i % (col_num * row_num)
        ir = int(math.floor(idx / col_num))
        ic = idx % col_num
        temp = clip_by_value(np.squeeze(images[i]), -1, 1)
        temp = (temp + 1) / 2 * 255
        temp = clip_by_value(np.round(temp), min=0, max=255)
        gLow = np.min(temp, axis=(0, 1, 2))
        gHigh = np.max(temp, axis=(0, 1, 2))
        temp = (temp - gLow) / (gHigh - gLow)
        cell_image[cell_id, (image_size+margin_syn)*ir:image_size + (image_size+margin_syn)*ir,
                    (image_size+margin_syn)*ic:image_size + (image_size+margin_syn)*ic,:] = temp
    return cell_image

def saveSampleResults(sample_results, filename, col_num=10, margin_syn=2):
    cell_image = img2cell(sample_results, col_num, col_num, margin_syn)
    scipy.misc.imsave(filename, np.squeeze(cell_image))
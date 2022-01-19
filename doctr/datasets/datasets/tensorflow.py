# Copyright (C) 2021-2022, Mindee.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import os
from typing import Any, List, Tuple

import tensorflow as tf

from doctr.io import read_img_as_tensor

from .base import _AbstractDataset, _VisionDataset

__all__ = ['AbstractDataset', 'VisionDataset']


class AbstractDataset(_AbstractDataset):

    def _read_sample(self, index: int) -> Tuple[tf.Tensor, Any]:
        img_name, target = self.data[index]
        # Read image
        img = read_img_as_tensor(os.path.join(self.root, img_name), dtype=tf.float32)
        target = [read_img_as_tensor(os.path.join(self.root, t), dtype=tf.float32) for t in target] # list of masks

        return img, tf.stack(target, axis=-1)

    @staticmethod
    def collate_fn(samples: List[Tuple[tf.Tensor, Any]]) -> Tuple[tf.Tensor, List[Any]]:

        images, targets = zip(*samples)
        images = tf.stack(images, axis=0)
        targets = tf.stack(targets, axis=0)

        return images, targets


class VisionDataset(AbstractDataset, _VisionDataset):
    pass

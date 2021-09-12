#!/usr/bin/env python

"""Manual test to check the values returned by an exported LPIPS model.
"""

from __future__ import print_function
import sys
import os
import numpy as np

sys.path.extend(
    [
        os.path.normpath(os.path.join(__file__, "..", "PerceptualSimilarity")),
    ]
)

RANDOM_SEED = 100

RUN_PYTORCH = False
RUN_TF2 = True

ARG_LPIPS = True
ARG_PRETRAINED = True
ARG_NET = "alex"

TF2_MODEL = "lpips_lin_alex.h5"

NUM_IMAGES = 7
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

# Generate the data
np.random.seed(RANDOM_SEED)
imgs_size = (NUM_IMAGES, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
imgs_a = np.random.uniform(0, 1, imgs_size).astype(np.float32)
imgs_b = np.random.uniform(0, 1, imgs_size).astype(np.float32)

if RUN_PYTORCH:
    import lpips
    import torch

    lpips_fn = lpips.LPIPS(lpips=ARG_LPIPS, pretrained=ARG_PRETRAINED, net=ARG_NET)

    # Prepare the data
    a = torch.from_numpy(np.transpose(imgs_a, [0, 3, 1, 2]) * 2.0 - 1.0)
    b = torch.from_numpy(np.transpose(imgs_b, [0, 3, 1, 2]) * 2.0 - 1.0)
    distance = lpips_fn(a, b)
    print("Pytorch distance result:", torch.squeeze(distance).detach().numpy())

if RUN_TF2:
    import tensorflow as tf

    lpips_fn = tf.keras.models.load_model(TF2_MODEL)
    distance = lpips_fn([imgs_a, imgs_b])
    print("TF2 distance result:", distance.numpy())

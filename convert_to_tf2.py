#!/usr/bin/env python

"""Convert the pytorch lpips metric to tensorflow 2.

Partly copied from https://github.com/alexlee-gk/lpips-tensorflow/blob/master/export_to_tensorflow.py
"""

from __future__ import print_function
import sys
import os
import argparse
import shutil

import onnx

sys.path.extend(
    [
        os.path.normpath(os.path.join(__file__, "..", "PerceptualSimilarity")),
        os.path.normpath(os.path.join(__file__, "..", "onnx2keras")),
    ]
)

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
ONNX_FILE = "model.onnx"
SAVED_MODEL_FOLDER = "model.savedmodel"


def init_torch_model(args):
    import lpips

    return lpips.LPIPS(
        lpips=not args.no_lpips, pretrained=not args.no_pretrained, net=args.net
    )


def torch_to_onnx(model):
    import torch

    dummy_im0 = torch.Tensor(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    dummy_im1 = torch.Tensor(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

    image_dynamic_axes = {0: "batch", 2: "height", 3: "width"}

    torch.onnx.export(
        model,
        (dummy_im0, dummy_im1),
        ONNX_FILE,
        input_names=["input_0", "input_1"],
        output_names=["distance"],
        dynamic_axes={
            "input_0": image_dynamic_axes,
            "input_1": image_dynamic_axes,
            "distance": {0: "batch"},
        },
    )


def onnx_to_keras():
    from onnx2keras import onnx_to_keras

    onnx_model = onnx.load(ONNX_FILE)
    return onnx_to_keras(
        onnx_model,
        ["input_0", "input_1"],
        input_shapes=[[3, None, None], [3, None, None]],
        verbose=False,
    )


def wrap_keras_in_out(model):
    import tensorflow as tf

    inp0 = tf.keras.Input((None, None, 3))
    inp1 = tf.keras.Input((None, None, 3))

    # NHWC to NCHW
    x0 = tf.transpose(inp0, [0, 3, 1, 2])
    x1 = tf.transpose(inp1, [0, 3, 1, 2])

    # Normalize from [0, 1] to [-1, 1]
    x0 = x0 * 2.0 - 1.0
    x1 = x1 * 2.0 - 1.0

    # Compute the distance using the saved_model
    distance = model([x0, x1])

    # Remove extra dimensions
    oup = tf.squeeze(distance, axis=[1, 2, 3])

    # Create a keras model for the complete script
    return tf.keras.Model(inputs=[inp0, inp1], outputs=[oup])


def main(args):
    torch_model = init_torch_model(args)
    torch_to_onnx(torch_model)
    keras_model = onnx_to_keras()
    keras_model = wrap_keras_in_out(keras_model)
    keras_model.save(args.output.name)

    # Cleanup
    os.remove(ONNX_FILE)
    shutil.rmtree(SAVED_MODEL_FOLDER)


def parse_args(arguments):
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--no-lpips",
        help="Do not use linear layers on top of base/trunk network.",
        action="store_true",
    )
    parser.add_argument(
        "--no-pretrained",
        help="Do not use pretrained layers.",
        action="store_true",
    )
    parser.add_argument(
        "--net",
        help="Base/trunk network.",
        default="alex",
        choices=["alex", "vgg", "squeeze"],
    )
    parser.add_argument(
        "output",
        help="Path to the output h5 model.",
        type=argparse.FileType("w"),
    )
    return parser.parse_args(arguments)


if __name__ == "__main__":
    sys.exit(main(parse_args(sys.argv[1:])))

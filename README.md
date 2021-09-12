# Converter script for the LPIPS metric from Pytorch to TF2 Keras

## Convert to a Keras h5 model

- Create the conda environment in `environment.yml` and activate it
- Run `python convert_to_tf2.py --net alex lpips_lin_alex.h5`
- Use the script `manual_test.py` to compare the outputs of the TF model to the output of the original code

Converted models can be donwloaded from the GitHub releases.

## Usage of the metric

Simple example
```python
import tensorflow as tf

lpips = tf.keras.models.load_model("lpips_lin_alex.h5")
img0 = tf.zeros([1, 120, 110, 3])
img1 = tf.zeros([1, 120, 110, 3])
distance = lpips([img0, img1])
print(distance.numpy())
```

This snipped can be copy-pasted in your script to define a `lpips` function.

```python
lpips_model = None

def lpips(imgs_a, imgs_b):
    """LPIPS: Learned Perceptual Image Patch Similarity metric

    R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang,
    “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,”
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, Jun. 2018, pp. 586–595.
    doi: 10.1109/CVPR.2018.00068.
    """
    if lpips_model is None:
        init_lpips_model()
    return lpips_model([imgs_a, imgs_b])


def init_lpips_model():
    global lpips_model
    model_file = tf.keras.utils.get_file(
        "lpips_lin_alex_0.1.0",
        "https://github.com/HedgehogCode/lpips-tf2/releases/download/0.1.0/lpips_lin_alex.h5",
        file_hash="d76a9756bf43f6b731a845968e8225ad",
        hash_algorithm="md5",
    )
    lpips_model = tf.keras.models.load_model(model_file)
```

## Notes

* The tf2 metric has to be executed on a GPU otherwise we get the error
  ```
  tensorflow.python.framework.errors_impl.UnimplementedError: The Conv2D op currently only supports the NHWC tensor format on the CPU. The op was given the format: NCHW [Op:Conv2D]
  ```
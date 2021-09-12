# Converter script for the LPIPS metric from Pytorch to TF2 Keras

## Convert to a Keras h5 model

- Create the conda environment in `environment.yml` and activate it
- Run `python convert_to_tf2.py --net alex lpips_lin_alex.h5`
- Use the script `manual_test.py` to compare the outputs of the TF model to the output of the original code

Converted models can be donwloaded from the GitHub releases.

## Usage of the metric

```
import tensorflow as tf

lpips = tf.keras.models.load_model("lpips_lin_alex.h5")
img0 = tf.zeros([1, 120, 110, 3])
img1 = tf.zeros([1, 120, 110, 3])
distance = lpips([img0, img1])
print(distance.numpy())
```

## Notes

* The tf2 metric has to be executed on a GPU otherwise we get the error
  ```
  tensorflow.python.framework.errors_impl.UnimplementedError: The Conv2D op currently only supports the NHWC tensor format on the CPU. The op was given the format: NCHW [Op:Conv2D]
  ```
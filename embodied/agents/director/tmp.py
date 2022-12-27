import tensorflow as tf
import numpy as np

class Input:

  def __init__(self, keys=['tensor'], dims=None):
    assert isinstance(keys, (list, tuple)), keys
    self._keys = tuple(keys)
    self._dims = dims or self._keys[0]

  def __call__(self, inputs):
    if not isinstance(inputs, dict):
      inputs = {'tensor': inputs}
    if not all(k in inputs for k in self._keys):
      needs = f'{{{", ".join(self._keys)}}}'
      found = f'{{{", ".join(inputs.keys())}}}'
      raise KeyError(f'Cannot find keys {needs} among inputs {found}.')
    values = [inputs[k] for k in self._keys]
    dims = len(inputs[self._dims].shape)
    for i, value in enumerate(values):
      if len(value.shape) > dims:
        values[i] = value.reshape(
            value.shape[:dims - 1] + [np.prod(value.shape[dims - 1:])])
    values = [x.astype(inputs[self._dims].dtype) for x in values]
    return tf.concat(values, -1)

if __name__ == '__main__':
    Input
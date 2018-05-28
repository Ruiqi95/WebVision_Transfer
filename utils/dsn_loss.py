# Cloned from tensorflow/models/research/domain_adaptation/domain_separation/losses.py 

# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Domain Adaptation Loss Functions.

The following domain adaptation loss functions are defined:

- Maximum Mean Discrepancy (MMD).
  Relevant paper:
    Gretton, Arthur, et al.,
    "A kernel two-sample test."
    The Journal of Machine Learning Research, 2012

- Correlation Loss on a batch.
"""
from functools import partial
import tensorflow as tf

#import grl_op_grads  # pylint: disable=unused-import
#import grl_op_shapes  # pylint: disable=unused-import
#import grl_ops
#grl_ops = tf.load_op_library('./_grl_ops.so')
import utils
slim = tf.contrib.slim


################################################################################
# SIMILARITY LOSS
################################################################################
def maximum_mean_discrepancy(x, y, kernel=utils.gaussian_kernel_matrix):
  r"""Computes the Maximum Mean Discrepancy (MMD) of two samples: x and y.

  Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of
  the distributions of x and y. Here we use the kernel two sample estimate
  using the empirical mean of the two distributions.

  MMD^2(P, Q) = || \E{\phi(x)} - \E{\phi(y)} ||^2
              = \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) },

  where K = <\phi(x), \phi(y)>,
    is the desired kernel function, in this case a radial basis kernel.

  Args:
      x: a tensor of shape [num_samples, num_features]
      y: a tensor of shape [num_samples, num_features]
      kernel: a function which computes the kernel in MMD. Defaults to the
              GaussianKernelMatrix.

  Returns:
      a scalar denoting the squared maximum mean discrepancy loss.
  """
  with tf.name_scope('MaximumMeanDiscrepancy'):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
  return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the MMD between two representations.

  This Maximum Mean Discrepancy (MMD) loss is calculated with a number of
  different Gaussian kernels.

  Args:
    source_samples: a tensor of shape [num_samples, num_features].
    target_samples: a tensor of shape [num_samples, num_features].
    weight: the weight of the MMD loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the MMD loss value.
  """
  sigmas = [
      1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
      1e3, 1e4, 1e5, 1e6
  ]
  gaussian_kernel = partial(
      utils.gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

  loss_value = maximum_mean_discrepancy(
      source_samples, target_samples, kernel=gaussian_kernel)
  loss_value = tf.maximum(1e-4, loss_value) * weight
  assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
  with tf.control_dependencies([assert_op]):
    tag = 'MMD Loss'
    if scope:
      tag = scope + tag
    tf.summary.scalar(tag, loss_value)
    tf.losses.add_loss(loss_value)

  return loss_value


def correlation_loss(source_samples, target_samples, weight, scope=None):
  """Adds a similarity loss term, the correlation between two representations.

  Args:
    source_samples: a tensor of shape [num_samples, num_features]
    target_samples: a tensor of shape [num_samples, num_features]
    weight: a scalar weight for the loss.
    scope: optional name scope for summary tags.

  Returns:
    a scalar tensor representing the correlation loss value.
  """
  with tf.name_scope('corr_loss'):
    source_samples -= tf.reduce_mean(source_samples, 0)
    target_samples -= tf.reduce_mean(target_samples, 0)

    source_samples = tf.nn.l2_normalize(source_samples, 1)
    target_samples = tf.nn.l2_normalize(target_samples, 1)

    source_cov = tf.matmul(tf.transpose(source_samples), source_samples)
    target_cov = tf.matmul(tf.transpose(target_samples), target_samples)

    corr_loss = tf.reduce_mean(tf.square(source_cov - target_cov)) * weight

  assert_op = tf.Assert(tf.is_finite(corr_loss), [corr_loss])
  with tf.control_dependencies([assert_op]):
    tag = 'Correlation Loss'
    if scope:
      tag = scope + tag
    tf.summary.scalar(tag, corr_loss)
    tf.losses.add_loss(corr_loss)

  return corr_loss

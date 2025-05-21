# Copyright (c) 2024 Rouast Labs
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
import numpy as np
from prpy.numpy.core import standardize, div0
from prpy.numpy.filters import detrend, moving_average
from prpy.numpy.physio import detrend_lambda_for_hr_response
from prpy.numpy.physio import moving_average_size_for_hr_response
from prpy.numpy.stride_tricks import window_view, reduce_window_view

from vitallens.enums import Mode
from vitallens.methods.simple_rppg_method import SimpleRPPGMethod

class POSRPPGMethod(SimpleRPPGMethod):
  """The POS algorithm by Wang et al. (2017)"""
  def __init__(
      self,
      config: dict,
      mode: Mode
    ):
    """Initialize the `POSRPPGMethod`
    
    Args:
      config: The configuration dict
      mode: The operation mode
    """
    super(POSRPPGMethod, self).__init__(config=config, mode=mode)
  def algorithm(
      self,
      rgb: np.ndarray,
      fps: float
    ) -> np.ndarray:
    """Use POS algorithm to estimate pulse from rgb signal.
    
    Args:
      rgb: The rgb signal. Shape (n_frames, 3)
      fps: The rate at which signal was sampled. Scalar
    Returns:
      sig: The estimated pulse signal. Shape (n_frames,)
    """
    # Create a windowed view into rgb
    window_length = self.est_window_length
    overlap = self.est_window_overlap
    # Check that enough frames are available
    if window_length > rgb.shape[0]:
      logging.warning("Too few frames available for POS method. Forcing shorter window.")
      window_length = rgb.shape[0] - rgb.shape[0] % 2
      overlap = window_length - 1
    # Create window view into rgb
    rgb_view, _, pad_end = window_view(
      x=rgb, min_window_size=window_length, max_window_size=window_length,
      overlap=overlap, pad_mode='constant', const_val=0)
    # Temporal normalization
    c_n = div0(rgb_view, np.nanmean(rgb_view, axis=1, keepdims=True), 0)
    # Projection
    s = np.matmul(c_n, np.asarray([[0, 1, -1], [-2, 1, 1]]).T)
    # Tuning
    sigma_1 = np.std(s[:,:,0], axis=-1, keepdims=True)
    sigma_2 = np.std(s[:,:,1], axis=-1, keepdims=True)
    h = s[:,:,0] + div0(sigma_1, sigma_2, fill=0) * s[:,:,1]
    # Reduce window view
    pos = reduce_window_view(h, overlap=overlap, pad_end=pad_end, hanning=False)
    # Invert
    pos = -1 * pos
    # Return
    return pos
  def pulse_filter(
      self, 
      sig: np.ndarray,
      fps: float
    ) -> np.ndarray:
    """Apply filters to the estimated pulse signal.

    Args:
      sig: The estimated pulse signal. Shape (n_frames,)
      fps: The rate at which signal was sampled. Scalar
    Returns:
      out: The filtered pulse signal. Shape (n_frames,)
    """
    # Detrend (high-pass equivalent)
    Lambda = detrend_lambda_for_hr_response(fps)
    sig = detrend(sig, Lambda)
    # Moving average (low-pass equivalent)
    size = moving_average_size_for_hr_response(fps)
    sig = moving_average(sig, size)
    # Standardize
    sig = standardize(sig)
    # Return
    return sig
  
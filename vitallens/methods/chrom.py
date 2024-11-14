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
from prpy.constants import SECONDS_PER_MINUTE
from prpy.numpy.signal import detrend, standardize, butter_bandpass, div0
from prpy.numpy.stride_tricks import window_view, reduce_window_view

from vitallens.enums import Mode
from vitallens.methods.simple_rppg_method import SimpleRPPGMethod
from vitallens.signal import detrend_lambda_for_hr_response

class CHROMRPPGMethod(SimpleRPPGMethod):
  """The CHROM algorithm by De Haan and Jeanne (2013)"""
  def __init__(
      self,
      config: dict,
      mode: Mode
    ):
    """Initialize the `CHROMRPPGMethod`
    
    Args:
      config: The configuration dict
      mode: The operation mode
    """
    super(CHROMRPPGMethod, self).__init__(config=config, mode=mode)
  def algorithm(
      self,
      rgb: np.ndarray,
      fps: float
    ) -> np.ndarray:
    """Use CHROM algorithm to estimate pulse from rgb signal.
    
    Args:
      rgb: The rgb signal. Shape (n_frames, 3)
      fps: The rate at which video was sampled. Scalar
    Returns:
      sig: The estimated pulse signal. Shape (n_frames,)
    """
    # Create a windowed view into rgb
    window_length = self.est_window_length
    overlap = self.est_window_overlap
    # Check that enough frames are available
    if window_length > rgb.shape[0]:
      logging.warning("Too few frames available for CHROM method. Forcing shorter window.")
      window_length = rgb.shape[0] - rgb.shape[0] % 2
      overlap = window_length - 1
    # Create window view into rgb
    rgb_view, _, pad_end = window_view(
      x=rgb, min_window_size=window_length, max_window_size=window_length,
      overlap=overlap, pad_mode='constant', const_val=0)
    # RGB norm
    rgb_n = div0(rgb_view, np.nanmean(rgb_view, axis=1, keepdims=True), 0) - 1
    # CHROM
    Xs = 3 * rgb_n[:,:,0] - 2 * rgb_n[:,:,1]
    Ys = (1.5 * rgb_n[:,:,0]) + rgb_n[:,:,1] - (1.5 * rgb_n[:,:,2])
    Xf = butter_bandpass(Xs, lowcut=40/SECONDS_PER_MINUTE, highcut=240/SECONDS_PER_MINUTE, fs=fps, axis=-1)
    Yf = butter_bandpass(Ys, lowcut=40/SECONDS_PER_MINUTE, highcut=240/SECONDS_PER_MINUTE, fs=fps, axis=-1)
    alpha = div0(np.std(Xf, axis=-1), np.std(Yf, axis=-1), fill=0)
    chrom = Xf - alpha[:,np.newaxis] * Yf
    # Reduce window view
    chrom = reduce_window_view(chrom, overlap=overlap, pad_end=pad_end, hanning=True)
    # Invert
    chrom = -1 * chrom
    # Return result
    return chrom
  def pulse_filter(
      self,
      sig: np.ndarray,
      fps: float
    ) -> np.ndarray:
    """Apply filters to the estimated pulse signal.

    Args:
      sig: The estimated pulse signal. Shape (n_frames,)
      fps: The rate at which video was sampled. Scalar
    Returns:
      x: The filtered pulse signal. Shape (n_frames,)
    """
    # Detrend (high-pass equivalent)
    Lambda = detrend_lambda_for_hr_response(fps)
    sig = detrend(sig, Lambda)
    # Standardize
    sig = standardize(sig)
    # Return
    return sig

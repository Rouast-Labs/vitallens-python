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

import numpy as np
from prpy.numpy.signal import detrend, moving_average, standardize

from vitallens.enums import Mode
from vitallens.methods.simple_rppg_method import SimpleRPPGMethod
from vitallens.signal import detrend_lambda_for_hr_response
from vitallens.signal import moving_average_size_for_hr_response

class GRPPGMethod(SimpleRPPGMethod):
  """The G algorithm by Verkruysse (2008)"""
  def __init__(
      self,
      config: dict,
      mode: Mode
    ):
    """Initialize the `GRPPGMethod`
    
    Args:
      config: The configuration dict
      mode: The operation mode
    """
    super(GRPPGMethod, self).__init__(config=config, mode=mode)
  def algorithm(
      self,
      rgb: np.ndarray,
      fps: float
    ) -> np.ndarray:
    """Use G algorithm to estimate pulse from rgb signal.

    Args:
      rgb: The rgb signal. Shape (n_frames, 3)
      fps: The rate at which signal was sampled.
    Returns:
      sig: The estimated pulse signal. Shape (n_frames,)
    """
    # Select green channel
    g = rgb[:,1]
    # Invert
    g = -1 * g
    # Return
    return g
  def pulse_filter(self, 
      sig: np.ndarray,
      fps: float
    ) -> np.ndarray:
    """Apply filters to the estimated pulse signal.

    Args:
      sig: The estimated pulse signal. Shape (n_frames,)
      fps: The rate at which signal was sampled.
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
  
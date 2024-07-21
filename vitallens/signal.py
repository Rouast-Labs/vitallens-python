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
from prpy.constants import SECONDS_PER_MINUTE
from prpy.numpy.signal import moving_average_size_for_response, estimate_freq
from prpy.numpy.stride_tricks import window_view, resolve_1d_window_view
from typing import Tuple, Union

from vitallens.constants import SECONDS_PER_MINUTE, CALC_HR_MAX, CALC_RR_MAX

def moving_average_size_for_hr_response(
    f_s: Union[float, int]
  ):
  return moving_average_size_for_response(f_s, CALC_HR_MAX / SECONDS_PER_MINUTE)

def moving_average_size_for_rr_response(
    f_s: Union[float, int]
  ):
  return moving_average_size_for_response(f_s, CALC_RR_MAX / SECONDS_PER_MINUTE)

def detrend_lambda_for_hr_response(
    f_s: Union[float, int]
  ):
  return int(0.1614*np.power(f_s, 1.9804))

def detrend_lambda_for_rr_response(
    f_s: Union[float, int]
  ):
  return int(4.4248*np.power(f_s, 2.1253))

def windowed_mean(
    x: np.ndarray,
    window_size: int,
    overlap: int
  ) -> np.ndarray:
  """Estimate the mean of an array using sliding windows. Returns same shape.
  
  Args:
    x: An array. Shape (n,)
    window_size: The size of the sliding window
    overlap: The overlap of subsequent locations of the sliding window
  Returns:
    out: The windowed mean. Shape (n,)
  """
  x = np.asarray(x)
  n = len(x)
  # Make sure there are enough vals
  if n <= window_size:
    raise ValueError("Not enough vals for frequency calculation.")
  else:
    # Generate a windowed view into x
    y, _, pad_end = window_view(
      x=x, min_window_size=window_size, max_window_size=window_size, overlap=overlap,
      pad_mode='reflect')
    # Estimate frequency for each window
    out = np.mean(y, axis=1)
    # Resolve to target dims
    out = resolve_1d_window_view(
      x=out, window_size=window_size, overlap=overlap, pad_end=pad_end, fill_method='start')
  # Make sure sizes match
  assert out.shape[0] == n, "out.shape[0] {} != {} n".format(
    out.shape[0], n)
  # Return
  return out

def windowed_freq(
    x: np.ndarray,
    window_size: int,
    overlap: int,
    f_s: Union[int, float],
    f_range: Tuple[Union[int, float], Union[int, float]] = None,
    f_res: Union[int, float] = None
  ) -> np.ndarray:
  """Estimate the varying frequency within a signal array using sliding windows. Returns same shape.
  
  Args:
    x: A signal with a frequency we want to estimate. Shape (n,)
    window_size: The size of the sliding window
    overlap: The overlap of subsequent locations of the sliding window
    f_s: The sampling frequency of x
    f_range: A range of (min, max) feasible frequencies to restrict the estimation to 
    f_res: The frequency resolution at which to estimate
  Returns:
    out: The estimated frequencies. Shape (n,)
  """
  x = np.asarray(x)
  n = len(x)
  # Make sure there are enough vals
  if n <= window_size:
    raise ValueError("Not enough vals for frequency calculation.")
  else:
    # Generate a windowed view into x
    y, _, pad_end = window_view(
      x=x, min_window_size=window_size, max_window_size=window_size, overlap=overlap,
      pad_mode='reflect')
    # Estimate frequency for each window
    freqs = estimate_freq(
      y, f_s=f_s, f_range=f_range, f_res=f_res, method='periodogram', axis=1)
    # Resolve to target dims
    freq_vals = resolve_1d_window_view(
      x=freqs, window_size=window_size, overlap=overlap, pad_end=pad_end, fill_method='start')
  # Make sure sizes match
  assert freq_vals.shape[0] == n, "freq_vals.shape[0] {} != {} n".format(
    freq_vals.shape[0], n)
  # Return
  return freq_vals

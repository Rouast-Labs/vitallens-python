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
from typing import Union
import warnings

class SignalBuffer:
  """A buffer for an arbitrary float signal.
  - Builds the buffer up over time.
  - Returns overlaps using mean.
  """
  def __init__(
      self,
      size: int,
      ndim: int,
      pad_val: float = 0,
      init_t: int = 0
    ):
    """Initialise the signal buffer.
    Args:
      size: The temporal length of the buffer. Signals are pushed through in FIFO order.
      ndim: The number of dims for the signal (e.g., ndim=1 means scalar signal with time dimension)
      pad_val: Constant scalar to pad empty time steps with
      init_t: Time for initialisation
    """
    self.size = size
    self.pad_val = pad_val
    self.min_increment = 1
    self.ndim = ndim
    # Buffer is a nested list with up to self.size lists
    # Each element is a tuple
    # - 0: t_start
    # - 1: t_end
    # - 2: signal  
    self.buffer = []
    self.t = init_t
    self.out = None
  def update(
      self,
      signal: Union[list, np.ndarray],
      dt: int
    ) -> np.ndarray:
    """Update the buffer with new signal series and amount of time steps passed.
    - Support arbitrary number of signal dims, but mostly intended for scalar series or one more dim
    (e.g., RGB signal over time).
    Args:
      signal: The signal. list or ndarray, shape (n_frames, dim1, dim2, ...)
      dt: The number of time steps passed. Scalar
    Returns:
      out: The signal of buffer size, with overlaps averaged. Shape (self.size, dim1, dim2, ...)
    """
    if isinstance(signal, list): signal = np.asarray(signal)
    if signal.size == 1 and self.ndim == 1: signal = np.full((dt,), fill_value=signal)
    assert isinstance(signal, np.ndarray), "signal should be np.ndarray but is {}".format(type(signal))
    assert len(signal.shape) == self.ndim
    assert len(signal) >= 1
    assert dt >= self.min_increment
    # Initialise self.out if necessary
    if self.out is None:
      self.out = np.empty((self.size, self.size,) + signal.shape[1:], signal.dtype)
    self.out[:] = np.nan
    # Update self.t
    self.t += dt
    # Update self.buffer
    self.buffer.append((self.t - signal.shape[0], self.t, signal))
    # Delete old buffer elements
    i = 0
    while i < len(self.buffer):
      if self.buffer[i][1] <= self.t - self.size:
        self.buffer.pop(0)
      else:
        i += 1
    return self.get()
  def get(self) -> np.ndarray:
    """Get the series of current buffer contents, with overlaps averaged.
    Returns:
      out: The signal of buffer size, with overlaps averaged. Shape (self.size, dim1, dim2, ...)
    """
    # No elements yet
    assert self.t > 0, "Update at least once before calling get()"
    # Assign buffer elements to self.out
    for i in range(len(self.buffer)):
      adj_t = self.t - self.size
      adj_t_start = self.buffer[i][0] - adj_t
      adj_t_end = self.buffer[i][1] - adj_t
      outside = 0 if adj_t_start >= 0 else abs(adj_t_start)
      self.out[i][adj_t_start+outside:adj_t_end] = self.buffer[i][2][outside:]
    # Reduce via mean (ignore warnings due to nan)
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=RuntimeWarning)
      result = np.nanmean(self.out, axis=0)
    # Replace np.nan with pad_val
    return np.nan_to_num(result, nan=self.pad_val)
  def clear(self):
    """Clear the contents"""
    self.buffer.clear()
    self.t = 0
    self.out = None

class MultiSignalBuffer:
  """Manages a dict of SignalBuffer instances"""
  def __init__(
      self,
      size: int,
      ndim: int,
      ignore_k: list,
      pad_val: float = 0
    ):
    """Initialise the multi signal buffer.
    Args:
      size: The temporal length of each buffer. Signals are pushed through in FIFO order.
      ndim: The number of dims for each signal (e.g., ndim=1 means scalar signal with time dimension)
      ignore_k: List of keys to ignore in update step
      pad_val: Constant scalar to pad empty time steps with
    """
    self.size = size
    self.ndim = ndim
    self.min_increment = 1
    self.ignore_k = ignore_k
    self.pad_val = pad_val
    self.signals = {}
    self.t = 0
  def update(
      self,
      signals: dict,
      dt: int
    ) -> dict:
    """Initialise or update each of the buffers corresponding to the entries in signals dict.
    Args:
      signals: Dictionary of signal updates. Each entry ndarray or list of shape (n_frames, dim1, dim2, ...)
      dt: The number of time steps passed. Scalar
    Returns:
      out: Dictionary of buffered signals, with overlaps averaged. Each entry of shape (self.size, dim1, dim2, ...)
    """
    assert isinstance(signals, dict)
    result = {}
    for k in signals:
      if k in self.ignore_k:
        continue
      if k not in self.signals:
        # Add k to self.signals
        self.signals[k] = SignalBuffer(
          size=self.size, ndim=self.ndim, pad_val=self.pad_val, init_t=self.t)
      # Run update
      result[k] = self.signals[k].update(signals[k], dt)
    self.t += dt
    return result
  def get(self) -> dict:
    """Get the series of current buffer contents, with overlaps averaged.
    Returns:
      out: Dictionary of buffered signals, with overlaps averaged. Each entry of shape (self.size, dim1, dim2, ...)
    """
    assert self.t > 0, "Update at least once before calling get()"
    result = {}
    for k in self.signals:
      if k in self.ignore_k:
        continue
      result[k] = self.signals[k].get()
    return result
  def clear(self):
    """Clear the contents"""
    for k in self.signals: 
      self.signals[k].clear()
    self.signals = {}
    
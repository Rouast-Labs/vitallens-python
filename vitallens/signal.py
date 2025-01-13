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

from vitallens.constants import CALC_HR_MIN, CALC_HR_MAX, CALC_RR_MIN, CALC_RR_MAX

def moving_average_size_for_hr_response(
    f_s: Union[float, int]
  ) -> int:
  """Get the moving average window size for a signal with HR information sampled at a given frequency
  
  Args:
    f_s: The sampling frequency
  Returns:
    The moving average size in number of signal vals
  """
  return moving_average_size_for_response(f_s, CALC_HR_MAX / SECONDS_PER_MINUTE)

def moving_average_size_for_rr_response(
    f_s: Union[float, int]
  ) -> int:
  """Get the moving average window size for a signal with RR information sampled at a given frequency
  
  Args:
    f_s: The sampling frequency
  Returns:
    The moving average size in number of signal vals
  """
  return moving_average_size_for_response(f_s, CALC_RR_MAX / SECONDS_PER_MINUTE)

def detrend_lambda_for_hr_response(
    f_s: Union[float, int]
  ) -> int:
  """Get the detrending lambda parameter for a signal with HR information sampled at a given frequency
  
  Args:
    f_s: The sampling frequency
  Returns:
    The lambda parameter
  """
  return int(0.1614*np.power(f_s, 1.9804))

def detrend_lambda_for_rr_response(
    f_s: Union[float, int]
  ) -> int:
  """Get the detrending lambda parameter for a signal with RR information sampled at a given frequency
  
  Args:
    f_s: The sampling frequency
  Returns:
    The lambda parameter
  """
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
  assert out.shape[0] == n, f"out.shape[0] {out.shape[0]} != {n} n"
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
  assert freq_vals.shape[0] == n, f"freq_vals.shape[0] {freq_vals.shape[0]} != {n} n"
  # Return
  return freq_vals

def reassemble_from_windows(
    x: np.ndarray,
    idxs: np.ndarray
  ) -> Tuple[np.ndarray, np.ndarray]:
  """Reassemble windowed data using corresponding idxs.

  Args:
    x: Data generated using a windowing operation. Shape (n_windows, n, window_size)
    idxs: Indices of x in the original 1-d array. Shape (n_windows, window_size)
  Returns:
    Tuple of
     - out: Reassembled data. Shape (n, n_idxs)
     - idxs: Reassembled idxs. Shape (n_idxs,)
  """
  x = np.asarray(x)
  idxs = np.asarray(idxs)
  # Transpose x (n, n_windows, window_size)
  x = np.transpose(x, (1, 0, 2))
  # Adjust indices based on their window position
  offset_idxs = idxs - np.arange(idxs.shape[0])[:, np.newaxis]
  # Find strictly increasing indices using np.maximum.accumulate
  flat_offset_idxs = offset_idxs.flatten()
  max_so_far = np.maximum.accumulate(flat_offset_idxs.flatten())
  mask = (flat_offset_idxs == max_so_far)  # Mask to keep only strictly increasing indices
  # Filter data based on mask and extract the final result values
  result = x.reshape(x.shape[0], -1)[:,mask]
  idxs = idxs.flatten()[mask]
  return result, idxs

def assemble_results(
    sig: np.ndarray,
    conf: np.ndarray,
    live: np.ndarray,
    fps: float,
    train_sig_names: list,
    pred_signals: list,
    method_name: str,
    can_provide_confidence: bool = True,
    min_t_hr: float = 2.,
    min_t_rr: float = 4.
  ) -> Tuple[dict, dict, dict, dict, np.ndarray]:
  """Assemble rPPG method results in the format expected by the API.
  Args:
    sig: The estimated signals. Shape (n_sig, n_frames)
    conf: The estimation confidence. Shape (n_sig, n_frames)
    live: The liveness confidence. Shape (n_frames,)
    fps: The sampling rate
    train_sig_names: The train signal names of the method
    pred_signals: The pred signals specs of the method
    method_name: The name of the method
    can_provide_confidence: Whether the method can provide a confidence estimate
    min_t_hr: Minimum amount of time signal to estimate hr
    min_t_rr: Minimum amount of time signal to estimate rr
  Returns:
    Tuple of
       - out_data: The estimated data/value for each signal.
       - out_unit: The estimation unit for each signal.
       - out_conf: The estimation confidence for each signal.
       - out_note: An explanatory note for each signal.
       - live: The face live confidence. Shape (n_frames,)
  """
  # Infer the signal length in seconds
  sig_t = sig.shape[1] / fps
  # Get the names of signals model outputs
  out_data, out_unit, out_conf, out_note = {}, {}, {}, {}
  confidence_note_scalar = ', along with a confidence level between 0 and 1.' if can_provide_confidence else '. This method is not capable of providing a confidence estimate, hence returning 1.'
  confidence_note_data = ', along with frame-wise confidences between 0 and 1.' if can_provide_confidence else '. This method is not capable of providing a confidence estimate, hence returning 1.'
  for name in pred_signals:
    if name == 'heart_rate' and 'ppg_waveform' in train_sig_names and sig_t > min_t_hr:
      ppg_ir_idx = train_sig_names.index('ppg_waveform')
      out_data[name] = estimate_freq(
        sig[ppg_ir_idx], f_s=fps, f_res=0.1/SECONDS_PER_MINUTE,
        f_range=(CALC_HR_MIN/SECONDS_PER_MINUTE, CALC_HR_MAX/SECONDS_PER_MINUTE),
        method='periodogram') * SECONDS_PER_MINUTE
      out_unit[name] = 'bpm'
      out_conf[name] = float(np.mean(conf[ppg_ir_idx]))
      out_note[name] = f'Estimate of the global heart rate using {method_name}{confidence_note_scalar}'
    elif name == 'respiratory_rate' and 'respiratory_waveform' in train_sig_names and sig_t > min_t_rr:
      resp_idx = train_sig_names.index('respiratory_waveform')
      out_data[name] = estimate_freq(
        sig[resp_idx], f_s=fps, f_res=0.1/SECONDS_PER_MINUTE,
        f_range=(CALC_RR_MIN/SECONDS_PER_MINUTE, CALC_RR_MAX/SECONDS_PER_MINUTE),
        method='periodogram') * SECONDS_PER_MINUTE
      out_unit[name] = 'bpm'
      out_conf[name] = float(np.mean(conf[resp_idx]))
      out_note[name] = f'Estimate of the global respiratory rate using {method_name}{confidence_note_scalar}'
    elif name == 'ppg_waveform':
      ppg_ir_idx = train_sig_names.index('ppg_waveform')
      out_data[name] = sig[ppg_ir_idx]
      out_unit[name] = 'unitless'
      out_conf[name] = conf[ppg_ir_idx]
      out_note[name] = f'Estimate of the ppg waveform using {method_name}{confidence_note_data}'
    elif name == 'respiratory_waveform':
      resp_idx = train_sig_names.index('respiratory_waveform')
      out_data[name] = sig[resp_idx]
      out_unit[name] = 'unitless'
      out_conf[name] = conf[resp_idx]
      out_note[name] = f'Estimate of the respiratory waveform using {method_name}{confidence_note_data}'
  return out_data, out_unit, out_conf, out_note, live

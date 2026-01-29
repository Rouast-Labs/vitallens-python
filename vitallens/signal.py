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
from prpy.numpy.physio import EScope, EMethod, HRVMetric
from prpy.numpy.physio import estimate_hr_from_signal, estimate_rr_from_signal
from prpy.numpy.physio import estimate_hrv_from_signal
from prpy.numpy.physio import CALC_HR_MIN_T, CALC_HRV_SDNN_MIN_T, CALC_RR_MIN_T
from prpy.numpy.physio import CALC_HRV_RMSSD_MIN_T, CALC_HRV_LFHF_MIN_T
from prpy.numpy.physio import CALC_HR_MAX_T, CALC_RR_MAX_T
from prpy.numpy.physio import CALC_HRV_SDNN_MAX_T, CALC_HRV_RMSSD_MAX_T, CALC_HRV_LFHF_MAX_T
from prpy.numpy.rolling import rolling_calc
from typing import Tuple

from vitallens.enums import Method

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
    method: Method,
    can_provide_confidence: bool = True,
    min_t_hr: float = CALC_HR_MIN_T,
    min_t_rr: float = CALC_RR_MIN_T,
    min_t_hrv_sdnn: float = CALC_HRV_SDNN_MIN_T,
    min_t_hrv_rmssd: float = CALC_HRV_RMSSD_MIN_T,
    min_t_hrv_lfhf: float = CALC_HRV_LFHF_MIN_T,
    hrv_conf_threshold: float = 0.5
  ) -> Tuple[dict, dict, dict, dict, np.ndarray]:
  """Assemble rPPG method results in the format expected by the API.
  Args:
    sig: The estimated signals. Shape (n_sig, n_frames)
    conf: The estimation confidence. Shape (n_sig, n_frames)
    live: The liveness confidence. Shape (n_frames,)
    fps: The sampling rate
    train_sig_names: The train signal names of the method
    pred_signals: The pred signals specs of the method
    method: The used method
    can_provide_confidence: Whether the method can provide a confidence estimate
    min_t_hr: Minimum amount of time signal to estimate hr
    min_t_rr: Minimum amount of time signal to estimate rr
    min_t_hrv_sdnn: Minimum amount of time signal to estimate hrv_sdnn
    min_t_hrv_rmssd: Minimum amount of time signal to estimate hrv_rmssd
    min_t_hrv_lfhf: Minimum amount of time signal to estimate hrv_lfhf
    hrv_conf_threshold: Peak detection confidence threshold for hrv estimation
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
  out_data, out_unit, out_conf, out_note = {}, {}, {}, {}
  confidence_note_scalar = ', along with a confidence level between 0 and 1.' if can_provide_confidence else '. This method is not capable of providing a confidence estimate, hence returning 1.'
  confidence_note_data = ', along with frame-wise confidences between 0 and 1.' if can_provide_confidence else '. This method is not capable of providing a confidence estimate, hence returning 1.'
  # Helper to get the method name string
  method_name = method.value if isinstance(method, Method) else str(method)
  # Helper function to process each vital sign
  def _process_vital(name, unit, vital_name_text, min_t, estimation_fn):
    if sig_t >= min_t:
      value, confidence = estimation_fn()
      if not np.isnan(value):
        out_data[name] = value
        out_conf[name] = confidence
        out_note[name] = f'Estimate of the global {vital_name_text} using {method_name}{confidence_note_scalar}'
      else:
        out_data[name], out_conf[name] = np.nan, np.nan
        out_note[name] = f'Estimate of the global {vital_name_text} using {method_name}. Too few values available to estimate.'
    else:
      out_data[name], out_conf[name] = np.nan, np.nan
      out_note[name] = f'Estimate of the global {vital_name_text} using {method_name}. Too few values available to estimate.'
    out_unit[name] = unit
  for name in pred_signals:
    if name == 'heart_rate' and 'ppg_waveform' in train_sig_names:
      ppg_idx = train_sig_names.index('ppg_waveform')
      def estimate_fn():
        hr = estimate_hr_from_signal(signal=sig[ppg_idx],
                                     f_s=fps,
                                     scope=EScope.GLOBAL,
                                     method=EMethod.PERIODOGRAM)
        conf_hr = float(np.mean(conf[ppg_idx]))
        return hr, conf_hr
      _process_vital(name, 'bpm', 'heart rate', min_t_hr, estimate_fn)
    elif name == 'respiratory_rate' and 'respiratory_waveform' in train_sig_names:
      resp_idx = train_sig_names.index('respiratory_waveform')
      def estimate_fn():
        rr = estimate_rr_from_signal(signal=sig[resp_idx],
                                     f_s=fps,
                                     scope=EScope.GLOBAL,
                                     method=EMethod.PERIODOGRAM)
        conf_rr = float(np.mean(conf[resp_idx]))
        return rr, conf_rr
      _process_vital(name, 'bpm', 'respiratory rate', min_t_rr, estimate_fn)
    elif name == 'ppg_waveform':
      ppg_idx = train_sig_names.index('ppg_waveform')
      out_data[name] = sig[ppg_idx]
      out_unit[name] = 'unitless'
      out_conf[name] = conf[ppg_idx]
      out_note[name] = f'Estimate of the ppg waveform using {method_name}{confidence_note_data}'
    elif name == 'respiratory_waveform':
      resp_idx = train_sig_names.index('respiratory_waveform')
      out_data[name] = sig[resp_idx]
      out_unit[name] = 'unitless'
      out_conf[name] = conf[resp_idx]
      out_note[name] = f'Estimate of the respiratory waveform using {method_name}{confidence_note_data}'
    elif 'hrv' in name and 'ppg_waveform' in train_sig_names:
      hrv_params = {
        'hrv_sdnn': ('ms', 'heart rate variability (SDNN)', min_t_hrv_sdnn, HRVMetric.SDNN),
        'hrv_rmssd': ('ms', 'heart rate variability (RMSSD)', min_t_hrv_rmssd, HRVMetric.RMSSD),
        'hrv_lfhf': ('unitless', 'heart rate variability (LF/HF)', min_t_hrv_lfhf, HRVMetric.LFHF)
      }
      if name in hrv_params:
        unit, text, min_t, metric = hrv_params[name]
        ppg_idx = train_sig_names.index('ppg_waveform')
        def estimate_fn():
          return estimate_hrv_from_signal(
            signal=sig[ppg_idx], f_s=fps, metric=metric,
            confidence=conf[ppg_idx], confidence_threshold=hrv_conf_threshold,
            min_window_size=int(fps*4), max_window_size=int(fps*8), overlap=int(fps*4),
            height=0, prominence=0.2, period_rel_tol=(0.5, 1.3),
            scope=EScope.GLOBAL, interp_skipped=True, min_dets=10, min_t=min_t
          )
        _process_vital(name, unit, text, min_t, estimate_fn)
  return out_data, out_unit, out_conf, out_note, live

def estimate_rolling_vitals(
    vital_signs_dict: dict,
    data: dict,
    conf: dict,
    signals_available: set,
    fps: float,
    video_duration_s: float
  ):
  """Helper to calculate and append rolling vitals to the results dictionary.
  
  Args:
    vital_signs_dict: The draft dict of vital signs to be modified
    data: The estimated data/value for each signal.
    conf: The estimation confidence for each signal.
    signals_available: The signals supported by the used rPPG method
    fps: The frame rate
    video_duration_s: The duration
  """
  # Helper function to process each rolling vital
  def _process_rolling_vital(name, unit, vital_name_text, data_series, conf_series):
    # Check if the entire series is NaN
    if np.all(np.isnan(data_series)):
      vital_signs_dict[name] = {
        'data': np.nan,
        'unit': unit,
        'confidence': np.nan,
        'note': f'Estimate of the rolling {vital_name_text} using VitalLens. Too few values available to estimate.'
      }
    else:
      vital_signs_dict[name] = {
        'data': data_series,
        'unit': unit,
        'confidence': conf_series,
        'note': f'Estimate of the rolling {vital_name_text} using VitalLens, along with frame-wise confidences between 0 and 1.'
      }
  try:
    if 'ppg_waveform' in signals_available and video_duration_s > CALC_HR_MAX_T:
      hr_window_size = int(CALC_HR_MAX_T*fps)
      hr_overlap = int(hr_window_size*7/8)
      rolling_hr = estimate_hr_from_signal(
        signal=data['ppg_waveform'], f_s=fps,
        window_size=hr_window_size, overlap=hr_overlap,
        scope=EScope.ROLLING, method=EMethod.PERIODOGRAM
      )
      rolling_conf_hr = rolling_calc(
        x=conf['ppg_waveform'], calc_fn=lambda x: np.nanmean(x, axis=-1),
        min_window_size=hr_window_size, max_window_size=hr_window_size, overlap=hr_overlap
      )
      _process_rolling_vital('rolling_heart_rate', 'bpm', 'heart rate', rolling_hr, rolling_conf_hr)
      hrv_signals = {'hrv_sdnn': (CALC_HRV_SDNN_MAX_T, HRVMetric.SDNN),
                     'hrv_rmssd': (CALC_HRV_RMSSD_MAX_T, HRVMetric.RMSSD),
                     'hrv_lfhf': (CALC_HRV_LFHF_MAX_T, HRVMetric.LFHF)}
      for hrv_name, (max_t, metric) in hrv_signals.items():
        if hrv_name in signals_available and video_duration_s > max_t:
          hrv_window_size = int(max_t*fps)
          hrv_overlap = int(hrv_window_size*7/8)
          rolling_hrv, _ = estimate_hrv_from_signal(
            signal=data['ppg_waveform'], f_s=fps, metric=metric,
            confidence=conf['ppg_waveform'], confidence_threshold=0.5,
            min_window_size=hrv_window_size, max_window_size=hrv_window_size,
            overlap=hrv_overlap, height=0, prominence=0.2, period_rel_tol=(0.5, 1.3),
            scope=EScope.ROLLING, interp_skipped=True, min_dets=10, min_t=max_t
          )
          rolling_hrv_conf = rolling_calc(
            x=conf['ppg_waveform'], calc_fn=lambda x: np.nanmin(x, axis=-1),
            min_window_size=hrv_window_size, max_window_size=hrv_window_size,
            overlap=hrv_overlap
          )
          unit = 'ms' if metric != HRVMetric.LFHF else 'unitless'
          _process_rolling_vital(f'rolling_{hrv_name}', unit, hrv_name, rolling_hrv, rolling_hrv_conf)
    if 'respiratory_waveform' in signals_available and video_duration_s > CALC_RR_MAX_T:
      rr_window_size = int(CALC_RR_MAX_T*fps)
      rr_overlap = int(rr_window_size*7/8)
      rolling_rr = estimate_rr_from_signal(
        signal=data['respiratory_waveform'], f_s=fps,
        window_size=rr_window_size, overlap=rr_overlap,
        scope=EScope.ROLLING, method=EMethod.PERIODOGRAM
      )
      rolling_conf_rr = rolling_calc(
        x=conf['respiratory_waveform'], calc_fn=lambda x: np.nanmean(x, axis=-1),
        min_window_size=rr_window_size, max_window_size=rr_window_size, overlap=rr_overlap
      )
      _process_rolling_vital('rolling_respiratory_rate', 'bpm', 'respiratory rate', rolling_rr, rolling_conf_rr)
  except ValueError as e:
    logging.info(f"Issue while computing rolling vitals: {e}")
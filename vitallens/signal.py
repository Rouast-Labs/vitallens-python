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
from prpy.numpy.physio import VITAL_REGISTRY, EScope
from prpy.numpy.rolling import rolling_calc
from typing import Tuple, Dict

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
    sig: Dict[str, np.ndarray],
    conf: Dict[str, np.ndarray],
    live: np.ndarray,
    fps: float,
    pred_signals: list,
    method_name: str,
    can_provide_confidence: bool = True
  ) -> Tuple[dict, dict, dict, dict, np.ndarray]:
  """Assemble rPPG method results in the format expected by the API.
  Args:
    sig: Dict of estimated signal arrays {name: array}. Each array shape (n_frames,)
    conf: Dict of estimation confidence arrays {name: array}. Each array shape (n_frames,)
    live: The liveness confidence. Shape (n_frames,)
    fps: The sampling rate
    pred_signals: The pred signals specs of the method
    method_name: The name of the used method
    can_provide_confidence: Whether the method can provide a confidence estimate
  Returns:
    Tuple of
       - out_data: The estimated data/value for each signal.
       - out_unit: The estimation unit for each signal.
       - out_conf: The estimation confidence for each signal.
       - out_note: An explanatory note for each signal.
       - live: The face live confidence. Shape (n_frames,)
  """
  # Infer the signal length in seconds
  sig_len = len(next(iter(sig.values()))) if sig else 0
  sig_t = sig_len / fps if fps > 0 else 0
  out_data, out_unit, out_conf, out_note = {}, {}, {}, {}
  # Standard confidence strings
  conf_txt_scalar = ', with confidence between 0 and 1.' if can_provide_confidence else '.'
  conf_txt_wave = ', with frame-wise confidence between 0 and 1.' if can_provide_confidence else '.'
  for name in pred_signals:
    meta = VITAL_REGISTRY.get(name)
    if not meta:
      continue
    vital_type = meta.get('type')
    unit = meta.get('unit', '')
    display_name = meta.get('display_name', name)
    aggregation = meta.get('aggregation')
    # --- CASE A: PROVIDED ---
    if vital_type == 'provided':
      if name in sig:
        if aggregation is None:
          # Waveform
          out_data[name] = sig[name]
          out_conf[name] = conf.get(name, np.zeros_like(sig[name]))
          out_note[name] = f"Estimate of the {display_name} using {method_name}{conf_txt_wave}"
        else:
          # Scalar (return aggregated global value)
          out_data[name] = np.nanmean(sig[name])
          out_conf[name] = np.nanmean(conf.get(name, 0))
          out_note[name] = f"Global estimate of {display_name} using {method_name}{conf_txt_scalar}"
        out_unit[name] = unit
      else:
        pass
    # --- CASE B: DERIVED (calculated locally) ---
    elif vital_type == 'derived':
      source = meta['source_signal']
      min_t = meta['min_t']
      calc_func = meta['func']
      if source in sig and sig_t >= min_t:
        try:
          # Execute the function defined in prpy
          val = calc_func(sig[source], fps, conf.get(source), scope=EScope.GLOBAL)
          if isinstance(val, tuple):
            val_est, conf_est = val
          else:
            val_est = val
            conf_src = conf.get(source, np.zeros_like(sig[source]))
            if aggregation == 'min':
              conf_est = np.nanmin(conf_src)
            else:
              conf_est = np.nanmean(conf_src)
          out_data[name] = val_est
          out_conf[name] = conf_est
          out_note[name] = f"Global estimate of {display_name} using {method_name}{conf_txt_scalar}"
        except Exception as e:
          logging.warning(f"Failed to derive {name}: {e}")
          out_data[name] = np.nan
          out_conf[name] = np.nan
          out_note[name] = f"Calculation error for {display_name}."
      else:
        out_data[name] = np.nan
        out_conf[name] = np.nan
        out_note[name] = f"Video too short ({sig_t:.1f}s) or signal too noisy to derive {display_name}."
      out_unit[name] = unit

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
  # Helper to standardize output format
  def _add_result(name, unit, display_name, val, c):
    if np.all(np.isnan(val)):
      vital_signs_dict[name] = {
        'data': np.nan, 'unit': unit, 'confidence': np.nan,
        'note': f'Video too short or signal too noisy for rolling {display_name}.'
      }
    else:
      vital_signs_dict[name] = {
        'data': val, 'unit': unit, 'confidence': c,
        'note': f'Rolling estimate of {display_name} with frame-wise confidence.'
      }

  target_vitals = signals_available.intersection(VITAL_REGISTRY.keys())
  for name in target_vitals:
    meta = VITAL_REGISTRY[name]
    # Validation checks
    if 'max_t' not in meta:
      continue
    max_t = meta['max_t']
    if video_duration_s <= max_t:
      continue
    v_type = meta.get('type')
    source_name = name if v_type == 'provided' else meta.get('source_signal')
    if source_name not in data:
      continue
    # Window parameters
    w_size = int(max_t * fps)
    overlap = int(w_size * 7 / 8)
    output_key = f"rolling_{name}"
    display_name = meta.get('display_name', name)
    val_roll = np.nan
    conf_roll = np.nan
    # Calculation
    try:
      if v_type == 'derived':
        calc_func = meta.get('func')
        res = calc_func(
          data[source_name], fps, conf.get(source_name),
          scope=EScope.ROLLING,
          window_size=w_size,
          overlap=overlap
        )
        val_roll = res[0] if isinstance(res, tuple) else res
      elif v_type == 'provided':
        val_roll = rolling_calc(
          x=data[source_name],
          calc_fn=np.nanmean,
          min_window_size=w_size, max_window_size=w_size, overlap=overlap
        )
    except Exception:
      continue
    # Confidence
    agg_func = np.nanmin if meta.get('aggregation') == 'min' else np.nanmean
    try:
      c_src = conf.get(source_name, np.zeros_like(data[source_name]))
      conf_roll = rolling_calc(
        x=c_src,
        calc_fn=lambda x: agg_func(x, axis=-1),
        min_window_size=w_size, max_window_size=w_size, overlap=overlap
      )
    except Exception:
      pass
    # Add to results
    _add_result(output_key, meta['unit'], display_name, val_roll, conf_roll)
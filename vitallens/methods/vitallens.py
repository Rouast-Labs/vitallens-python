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

import base64
import concurrent.futures
import math
import numpy as np
from prpy.numpy.face import get_roi_from_det
from prpy.numpy.signal import detrend, moving_average, standardize
from prpy.numpy.signal import interpolate_cubic_spline, estimate_freq
import json
import logging
import requests
from typing import Union, Tuple

from vitallens.constants import API_MAX_FRAMES, API_URL, API_OVERLAP
from vitallens.constants import SECONDS_PER_MINUTE, CALC_HR_MIN, CALC_HR_MAX, CALC_RR_MIN, CALC_RR_MAX
from vitallens.errors import VitalLensAPIKeyError, VitalLensAPIQuotaExceededError, VitalLensAPIError
from vitallens.methods.rppg_method import RPPGMethod
from vitallens.signal import detrend_lambda_for_hr_response, detrend_lambda_for_rr_response
from vitallens.signal import moving_average_size_for_hr_response, moving_average_size_for_rr_response
from vitallens.utils import probe_video_inputs, parse_video_inputs

class VitalLensRPPGMethod(RPPGMethod):
  def __init__(
      self,
      config: dict,
      api_key: str
    ):
    super(VitalLensRPPGMethod, self).__init__(config=config)
    self.api_key = api_key
    self.input_size = config['input_size']
    self.roi_method = config['roi_method']
    self.signals = config['signals']
  def __call__(
      self,
      frames: Union[np.ndarray, str],
      faces: np.ndarray,
      fps: float = None,
      override_fps_target: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate vitals from video frames using the VitalLens API.

    Args:
      frames: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        in unscaled uint8 RGB format, or a path to a video file.
      faces: The face detection boxes as np.int64. Shape (n_frames, 4) in form (x0, y0, x1, y1)
      fps: The rate at which video was sampled.
      override_fps_target: Override the method's default inference fps (optional).
    Returns:
      sig: Estimated pulse signal. Shape (n_sig, n_frames)
      conf: Estimation confidence. Shape (n_sig, n_frames)
      live: Liveness estimation. Shape (n_frames,)
    """
    inputs_shape, fps = probe_video_inputs(video=frames, fps=fps)
    # Choose representative face detection
    face = faces[np.argmin(np.linalg.norm(faces - np.median(faces, axis=0), axis=1))]
    roi = get_roi_from_det(
      face, roi_method=self.roi_method, clip_dims=(inputs_shape[2], inputs_shape[1]))
    if np.any(np.logical_or(
      (faces[:,2] - faces[:,0]) * 0.5 < np.maximum(0, faces[:,0] - roi[0]) + np.maximum(0, faces[:,2] - roi[2]),
      (faces[:,3] - faces[:,1]) * 0.5 < np.maximum(0, faces[:,1] - roi[1]) + np.maximum(0, faces[:,3] - roi[3]))):
      logging.warn("Large face movement detected")
    # Parse the inputs
    logging.debug("Preparing video for inference...")
    frames_ds, fps, inputs_shape, ds_factor, _ = parse_video_inputs(
      video=frames, fps=fps, target_size=self.input_size, roi=roi,
      target_fps=override_fps_target if override_fps_target is not None else self.fps_target,
      library='prpy', scale_algorithm='bilinear')
    # Check the number of frames to be processed
    ds_len = frames_ds.shape[0]
    if ds_len <= API_MAX_FRAMES:
      # API supports up to MAX_FRAMES at once - process all frames
      sig_ds, conf_ds, live_ds = self.process_api(frames_ds)
    else:
      # Longer videos are split up with small overlaps
      ds_len = frames_ds.shape[0]
      n_splits = math.ceil((ds_len - API_MAX_FRAMES) / (API_MAX_FRAMES - API_OVERLAP)) + 1
      split_len = math.ceil((ds_len + (n_splits-1) * API_OVERLAP) / n_splits)
      start_idxs = [i for i in range(0, ds_len - n_splits * API_OVERLAP, split_len - API_OVERLAP)]
      end_idxs = [min(i + split_len, ds_len) for i in start_idxs]
      logging.info("Running inference for {} frames using {} requests...".format(ds_len, n_splits))
      # Process the splits in parallel
      with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda i: self.process_api(frames_ds[start_idxs[i]:end_idxs[i]]), range(n_splits)))
      # Aggregate the results
      sig_results, conf_results, live_results = zip(*results)
      sig_ds = np.concatenate([sig_results[0]] + [[x[API_OVERLAP:] for x in e] for e in sig_results[1:]], axis=-1)
      conf_ds = np.concatenate([conf_results[0]] + [[x[API_OVERLAP:] for x in e] for e in conf_results[1:]], axis=-1)
      live_ds = np.concatenate([live_results[0]] + [x[API_OVERLAP:] for x in live_results[1:]], axis=-1)      
    # Interpolate to original sampling rate (n_frames,)
    sig = interpolate_cubic_spline(
      x=np.arange(inputs_shape[0])[0::ds_factor], y=sig_ds, xs=np.arange(inputs_shape[0]), axis=1)
    conf = interpolate_cubic_spline(
      x=np.arange(inputs_shape[0])[0::ds_factor], y=conf_ds, xs=np.arange(inputs_shape[0]), axis=1)
    live = interpolate_cubic_spline(
      x=np.arange(inputs_shape[0])[0::ds_factor], y=live_ds, xs=np.arange(inputs_shape[0]), axis=0)
    # Filter (n_frames,)
    sig = np.asarray([self.postprocess(p, fps, type=name) for p, name in zip(sig, ['ppg', 'resp'])])
    # Estimate summary vitals
    hr = estimate_freq(
      sig[0], f_s=fps, f_res=0.1/SECONDS_PER_MINUTE,
      f_range=(CALC_HR_MIN/SECONDS_PER_MINUTE, CALC_HR_MAX/SECONDS_PER_MINUTE),
      method='periodogram') * SECONDS_PER_MINUTE
    rr = estimate_freq(
      sig[1], f_s=fps, f_res=0.1/SECONDS_PER_MINUTE,
      f_range=(CALC_RR_MIN/SECONDS_PER_MINUTE, CALC_RR_MAX/SECONDS_PER_MINUTE),
      method='periodogram') * SECONDS_PER_MINUTE
    # Confidences
    hr_conf = float(np.mean(conf[0]))
    rr_conf = float(np.mean(conf[1]))
    # Assemble results
    out_data, out_unit, out_conf, out_note = {}, {}, {}, {}
    for name in self.signals:
      if name == 'heart_rate':
        out_data[name] = hr
        out_unit[name] = 'bpm'
        out_conf[name] = hr_conf
        out_note[name] = 'Estimate of the heart rate using VitalLens, along with a confidence level between 0 and 1.'
      elif name == 'respiratory_rate':
        out_data[name] = rr
        out_unit[name] = 'bpm'
        out_conf[name] = rr_conf
        out_note[name] = 'Estimate of the respiratory rate using VitalLens, along with a confidence level between 0 and 1.'
      elif name == 'ppg_waveform':
        out_data[name] = sig[0]
        out_unit[name] = 'unitless'
        out_conf[name] = conf[0]
        out_note[name] = 'Estimate of the ppg waveform using VitalLens, along with a frame-wise confidences between 0 and 1.'
      elif name == 'respiratory_waveform':
        out_data[name] = sig[1]
        out_unit[name] = 'unitless'
        out_conf[name] = conf[1]
        out_note[name] = 'Estimate of the respiratory waveform using VitalLens, along with a frame-wise confidences between 0 and 1.'
    return out_data, out_unit, out_conf, out_note, live
  def process_api(
      self,
      frames: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process frames with the VitalLens API.

    Args:
      frames: The video frames. Shape (n_frames<=MAX_FRAMES, h==input_size, w==input_size, 3)
    Returns:
      sig: Estimated signals. Shape (n_sig, n_frames)
      conf: Estimation confidences. Shape (n_sig, n_frames)
      live: Liveness estimation. Shape (n_frames,)
    """
    assert frames.shape[0] <= API_MAX_FRAMES
    # Prepare API header and payload
    headers = {"x-api-key": self.api_key}
    payload = {"video": base64.b64encode(frames.tobytes()).decode('utf-8')}
    # Ask API to process video
    response = requests.post(API_URL, headers=headers, json=payload)
    response_body = json.loads(response.text)
    # Check if call was successful
    if response.status_code != 200:
      logging.error("Error {}: {}".format(response.status_code, response_body['message']))
      if response.status_code == 403:
        raise VitalLensAPIKeyError()
      elif response.status_code == 429:
        raise VitalLensAPIQuotaExceededError()
      elif response.status_code == 400:
        raise VitalLensAPIError("Error occurred in the API. Message: {}".format(response_body['message']))
      else:
        raise Exception("Error {}: {}".format(response.status_code, response_body['message']))
    # Parse response
    sig_ds = np.stack([
      np.asarray(response_body["vital_signs"]["ppg_waveform"]["data"]),
      np.asarray(response_body["vital_signs"]["respiratory_waveform"]["data"]),
    ], axis=0)
    conf_ds = np.stack([
      np.asarray(response_body["vital_signs"]["ppg_waveform"]["confidence"]),
      np.asarray(response_body["vital_signs"]["respiratory_waveform"]["confidence"]),
    ], axis=0)
    live_ds = np.asarray(response_body["face"]["confidence"])
    return sig_ds, conf_ds, live_ds
  def postprocess(self, sig, fps, type='ppg', filter=True):
    """Apply filters to the estimated signal. 
    Args:
      sig: The estimated signal. Shape (n_frames,)
      fps: The rate at which video was sampled. Scalar
      type: The signal type - either 'pulse' or 'resp'.
      filter: Whether to apply filters
    Returns:
      x: The filtered signal. Shape (n_frames,)
    """
    n_frames = sig.shape[-1]
    if filter:
      # Get filter parameters
      if type == 'ppg':
        size = moving_average_size_for_hr_response(fps)
        Lambda = detrend_lambda_for_hr_response(fps)
      elif type == 'resp':
        size = moving_average_size_for_rr_response(fps)
        Lambda = detrend_lambda_for_rr_response(fps)
      else:
        raise ValueError("Type {} not implemented!".format(type))
      if sig.shape[-1] < 4 * API_MAX_FRAMES:
        # Detrend only for shorter videos for performance reasons
        sig = detrend(sig, Lambda)
      # Moving average
      sig = moving_average(sig, size)
      # Standardize
      sig = standardize(sig)
    # Return
    assert sig.shape == (n_frames,)
    return sig

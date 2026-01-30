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
from prpy.numpy.core import standardize
from prpy.numpy.face import get_roi_from_det
from prpy.numpy.filters import detrend, moving_average
from prpy.numpy.image import probe_image_inputs, parse_image_inputs
from prpy.numpy.interp import interpolate_filtered
from prpy.numpy.physio import detrend_lambda_for_hr_response, detrend_lambda_for_rr_response
from prpy.numpy.physio import moving_average_size_for_hr_response, moving_average_size_for_rr_response
from prpy.numpy.physio import API_KEY_MAP
from prpy.numpy.utils import enough_memory_for_ndarray
import json
import logging
import requests
import os
from typing import Union, Tuple

from vitallens.constants import API_MAX_FRAMES, API_URL, API_OVERLAP, API_RESOLVE_URL
from vitallens.enums import Mode
from vitallens.errors import VitalLensAPIKeyError, VitalLensAPIQuotaExceededError, VitalLensAPIError
from vitallens.methods.rppg_method import RPPGMethod
from vitallens.signal import reassemble_from_windows, assemble_results
from vitallens.utils import check_faces_in_roi

def _resolve_model_config(
    api_key: str,
    model_name: str,
    proxies: dict = None
  ) -> dict:
  """Calls the /resolve-model endpoint to get the correct model config.
  
  Args:
    api_key: The API key
    model_name: The requested model name (e.g., 'vitallens', 'vitallens-2.0')
    proxies: Dictionary mapping protocol to the URL of the proxy
  Returns:
    resolved_config: The config of the resolved model 
  """
  headers = {}
  if api_key:
    headers["x-api-key"] = api_key
  params = {}
  if model_name != "vitallens":
    params['model'] = model_name
  try:
    response = requests.get(API_RESOLVE_URL, headers=headers, params=params, proxies=proxies)
    response.raise_for_status()
    data = response.json()
    # Prepare a clean config dictionary from the API response
    resolved_config = data['config']
    resolved_config['model'] = data['resolved_model']
    if 'supported_vitals' in resolved_config:
      vitals = resolved_config.pop('supported_vitals')
      resolved_config['signals'] = {API_KEY_MAP.get(c, c) for c in vitals}
    return resolved_config
  except requests.exceptions.HTTPError as e:
    error_msg = f"Failed to resolve model config: Status {e.response.status_code}"
    try:
      error_details = e.response.json().get('message', e.response.text)
      error_msg += f" - {error_details}"
    except json.JSONDecodeError:
      error_msg += f" - {e.response.text}"

    if e.response.status_code in [401, 403]:
      raise VitalLensAPIKeyError(error_msg)
    else:
      raise VitalLensAPIError(error_msg)
  except Exception as e:
    raise VitalLensAPIError(f"An unexpected error occurred while resolving model config: {e}")

class VitalLensRPPGMethod(RPPGMethod):
  """RPPG method using the VitalLens API for inference"""
  def __init__(
      self,
      mode: Mode,
      api_key: str,
      requested_model_name: str,
      proxies: dict = None
    ):
    """Initialize the `VitalLensRPPGMethod`
    
    Args:
      mode: The operation mode
      api_key: The API key
      requested_model_name: The requested VitalLens model name,
      proxies: Dictionary mapping protocol to the URL of the proxy
    """
    super(VitalLensRPPGMethod, self).__init__(mode=mode)
    
    if proxies is None and (api_key is None or api_key == ''):
      raise VitalLensAPIKeyError()
    self.api_key = api_key
    self.proxies = proxies

    # Resolve model config
    resolved_config = _resolve_model_config(api_key=api_key, model_name=requested_model_name, proxies=proxies)
    self.parse_config(resolved_config)

    self.resolved_model = resolved_config['model']
    self.requested_model_name = requested_model_name if requested_model_name != "vitallens" else None

    if mode == Mode.BURST:
      self.state = None
      self.input_buffer = None

  def parse_config(self, config: dict):
    """Set properties based on the config.
    
    Args:
      config: The method's config dict
    """
    super(VitalLensRPPGMethod, self).parse_config(config=config)
    self.n_inputs = int(config['n_inputs'])
    self.input_size = int(config['input_size'])
    self.signals = config.get('signals', set())
    self.est_window_length = 0
    self.est_window_overlap = 0

  def __call__(
      self,
      inputs: Union[np.ndarray, str],
      faces: np.ndarray,
      fps: float = None,
      override_fps_target: float = None,
      override_global_parse: bool = None
    ) -> Tuple[dict, dict, dict, dict, np.ndarray]:
    """Estimate vitals from video frames using the VitalLens API.

    Args:
      inputs: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        in unscaled uint8 RGB format, or a path to a video file.
      faces: The face detection boxes as np.int64. Shape (n_frames, 4) in form (x0, y0, x1, y1)
      fps: The rate at which video was sampled.
      override_fps_target: Override the method's default inference fps (optional).
      override_global_parse: If True, always use global parse. If False, don't use global parse.
        If None, choose based on video.
    Returns:
      Tuple of
       - out_data: The estimated data/value for each signal.
       - out_unit: The estimation unit for each signal.
       - out_conf: The estimation confidence for each signal.
       - out_note: An explanatory note for each signal.
       - live: The face live confidence. Shape (n_frames,)
    """
    inputs_shape, fps, video_issues = probe_image_inputs(inputs, fps=fps)
    # Check the number of frames to be processed
    inputs_n = inputs_shape[0]
    fps_target = override_fps_target if override_fps_target is not None else self.fps_target
    expected_ds_factor = max(round(fps / fps_target), 1)
    expected_ds_n = math.ceil(inputs_n / expected_ds_factor)
    # Check if we should parse the video globally
    video_fits_in_memory = enough_memory_for_ndarray(
      shape=(expected_ds_n, self.input_size, self.input_size, 3), dtype=np.uint8,
      max_fraction_of_available_memory_to_use=0.1)
    global_face = faces[np.argmin(np.linalg.norm(faces - np.median(faces, axis=0), axis=1))]
    global_roi = get_roi_from_det(
      global_face, roi_method=self.roi_method, clip_dims=(inputs_shape[2], inputs_shape[1]), detector='ultralight-rfb')
    global_faces_in_roi = check_faces_in_roi(faces=faces, roi=global_roi, percentage_required_inside_roi=(0.6, 1.0))
    global_parse = isinstance(inputs, str) and video_fits_in_memory and (video_issues or global_faces_in_roi)
    if override_global_parse is not None: global_parse = override_global_parse
    if global_parse:
      # Parse entire video for inference globally
      frames, _, _, _, idxs = parse_image_inputs(
        inputs=inputs, fps=fps, roi=global_roi, target_size=self.input_size, target_fps=fps_target,
        preserve_aspect_ratio=False, library='prpy', scale_algorithm='bilinear', 
        trim=None, allow_image=False, videodims=True)
    else:
      frames = inputs
    # Longer videos are split up with small overlaps
    n_splits = 1 if expected_ds_n <= API_MAX_FRAMES else math.ceil((expected_ds_n - API_MAX_FRAMES) / (API_MAX_FRAMES - API_OVERLAP)) + 1
    split_len = inputs_n if n_splits == 1 else math.ceil((inputs_n + (n_splits-1) * API_OVERLAP) / n_splits)
    start_idxs = [i * (split_len - API_OVERLAP) for i in range(n_splits)]
    end_idxs = [min(start + split_len, inputs_n) for start in start_idxs]
    start_idxs = [max(0, end - split_len) for end in end_idxs]
    logging.info(
      f"Analyzing video of {expected_ds_n} frames using {n_splits} request{'s' if n_splits > 1 else ''}. "
      f"Using {n_splits * split_len} frames in total{' including overlaps' if n_splits > 1 else ''}...")
    # Process the batches in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
      results = list(executor.map(lambda i: self.process_api_batch(
        batch=i, n_batches=n_splits, inputs=frames, inputs_shape=inputs_shape,
        faces=faces, fps_target=fps_target, fps=fps, global_parse=global_parse,
        start=None if n_splits == 1 else start_idxs[i],
        end=None if n_splits == 1 else end_idxs[i]), range(n_splits)))
    # Transpose results (list of tuples -> tuple of lists)
    sig_results, conf_results, live_results, idxs_results = zip(*results)
    # Aggregate sig estimates
    keys = sorted(list(sig_results[0].keys())) if sig_results else []
    if keys:
      def stack_dicts(dict_list):
        return np.array([[b.get(k, np.full(len(idxs_results[i]), np.nan)) for k in keys] 
                        for i, b in enumerate(dict_list)])
      sig_tensor = stack_dicts(sig_results)
      conf_tensor = stack_dicts(conf_results)
      # Returns (n_channels, total_valid_frames)
      rec_sig, idxs = reassemble_from_windows(x=sig_tensor, idxs=idxs_results)
      rec_conf, _ = reassemble_from_windows(x=conf_tensor, idxs=idxs_results)
      # Interpolate to original sampling rate (n_channels, inputs_n)
      sig = interpolate_filtered(t_in=idxs, s_in=rec_sig, t_out=np.arange(inputs_n), axis=1, extrapolate=True)
      conf = interpolate_filtered(t_in=idxs, s_in=rec_conf, t_out=np.arange(inputs_n), axis=1, extrapolate=True)
      # Unpack
      sig_ds = dict(zip(keys, sig))
      conf_ds = dict(zip(keys, conf))
    else:
      sig_ds, conf_ds = {}, {}
    # Aggregate liveness
    live_stacked = np.array(live_results)[:, np.newaxis, :]
    rec_live, _ = reassemble_from_windows(x=live_stacked, idxs=idxs_results)
    live = interpolate_filtered(t_in=idxs, s_in=rec_live[0], t_out=np.arange(inputs_n), axis=0, extrapolate=True)
    # Postprocess
    if self.op_mode == Mode.BATCH:
      for key in sig_ds:
        if 'waveform' in key:
          sig_ds[key] = self.postprocess(sig_ds[key], fps, type=key)
    # Assemble and return the results
    return assemble_results(
      sig=sig_ds,
      conf=conf_ds,
      live=live,
      fps=fps,
      pred_signals=list(self.signals),
      method_name=self.resolved_model,
      can_provide_confidence=True
    )

  def process_api_batch(
      self,
      batch: int,
      n_batches: int,
      inputs: Tuple[np.ndarray, str],
      inputs_shape: tuple,
      faces: np.ndarray,
      fps_target: float,
      start: int = None,
      end: int = None,
      fps: float = None,
      global_parse: bool = False
    ) -> Tuple[dict, dict, np.ndarray, np.ndarray]:
    """Process a batch of frames with the VitalLens API.

    Args:
      batch: The number of this batch.
      n_batches: The total number of batches.
      inputs: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        with a sequence of frames in unscaled uint8 RGB format, or a path to a video file.
      inputs_shape: The original shape of the inputs.
      faces: The face detection boxes as np.int64. Shape (n_frames, 4) in form (x0, y0, x1, y1)
      fps_target: The target frame rate at which to run inference.
      start: The index of first frame of the video to analyze in this batch.
      end: The index of the last frame of the video to analyze in this batch.
      fps: The frame rate of the input video. Required if type(video) == np.ndarray
      global_parse: Flag that indicates whether video has already been parsed.
    Returns:
      Tuple of
       - sig: Dict of estimated signals {name: array}. Each array has shape (n_frames,).
       - conf: Dict of estimation confidences {name: array}. Each array has shape (n_frames,).
       - live: Liveness estimation. Shape (n_frames,)
       - idxs: Indices in inputs that were processed. Shape (n_frames)
    """
    logging.debug(f"Batch {batch}/{n_batches}...")
    # Trim face detections to batch if necessary
    if start is not None and end is not None:
      faces = faces[start:end]
    # Choose representative face detection
    face = faces[np.argmin(np.linalg.norm(faces - np.median(faces, axis=0), axis=1))]
    roi = get_roi_from_det(
      face, roi_method=self.roi_method, clip_dims=(inputs_shape[2], inputs_shape[1]))
    if not check_faces_in_roi(faces=faces, roi=roi):
      logging.warning("Large face movement detected")
    if global_parse:
      # Inputs have already been parsed globally.
      assert isinstance(inputs, np.ndarray)
      frames_ds = inputs
      ds_factor = math.ceil(inputs_shape[0] / frames_ds.shape[0])
      # Trim frames to batch if necessary
      if start is not None and end is not None:
        start_ds = start // ds_factor
        end_ds = math.ceil((end-start)/ds_factor) + start_ds
        frames_ds = frames_ds[start_ds:end_ds]
        idxs = list(range(start, end, ds_factor))
      else:
        idxs = list(range(0, inputs_shape[0], ds_factor))
    else:
      # Buffer inputs for burst mode
      if self.op_mode == Mode.BURST:
        # Inputs in burst mode are always np.ndarray
        if self.state is not None:
          # State has been initialized
          assert self.input_buffer is not None
          if inputs.shape[1:] != self.input_buffer.shape[1:]:
            raise ValueError("In burst mode, input dimensions must be consistent.")
          inputs = np.concatenate([self.input_buffer, inputs], axis=0)
        self.input_buffer = inputs[-(self.n_inputs-1):]
      # Inputs have not been parsed globally. Parse the inputs
      frames_ds, _, _, ds_factor, idxs = parse_image_inputs(
        inputs=inputs, fps=fps, roi=roi, target_size=self.input_size, target_fps=fps_target,
        preserve_aspect_ratio=False, library='prpy', scale_algorithm='bilinear', 
        trim=(start, end) if start is not None and end is not None else None,
        allow_image=False, videodims=True)
    # Make sure we have the correct number of frames
    idxs = np.asarray(idxs)
    expected_n = math.ceil(((end-start) if start is not None and end is not None else inputs_shape[0]) / ds_factor)
    if (self.op_mode == Mode.BURST and self.state is not None): expected_n += (self.n_inputs - 1)
    if frames_ds.shape[0] != expected_n or idxs.shape[0] != expected_n:
      raise ValueError("Unexpected number of frames returned. Try to set `override_global_parse` to `True` or `False`.")
    # Prepare API header and payload
    headers = {}
    if self.api_key:
      headers["x-api-key"] = self.api_key
    origin = os.getenv('VITALLENS_API_ORIGIN', 'vitallens-python')
    payload = {"video": base64.b64encode(frames_ds.tobytes()).decode('utf-8'), "origin": origin}
    if self.requested_model_name:
      payload['model'] = self.requested_model_name
    if self.op_mode == Mode.BURST and self.state is not None:
      # State and frame buffer have been initialized
      assert self.input_buffer is not None
      payload["state"] = base64.b64encode(self.state.astype(np.float32).tobytes()).decode('utf-8')
      # Adjust idxs
      idxs = idxs[(self.n_inputs-1):] - (self.n_inputs-1)
      logging.debug(f"Providing state, which means that {self.n_inputs-1} less frames will be used and results for {self.n_inputs-1} less frames will be returned.")
    # Ask API to process video
    response = requests.post(API_URL, headers=headers, json=payload, proxies=self.proxies)
    response_body = json.loads(response.text)
    # Check if call was successful
    if response.status_code != 200:
      logging.error(f"Error {response.status_code}: {response_body['message']}")
      if response.status_code == 403:
        raise VitalLensAPIKeyError()
      elif response.status_code == 429:
        raise VitalLensAPIQuotaExceededError()
      elif response.status_code in [400, 422, 500]:
        raise VitalLensAPIError(f"API Error: {response_body['message']}")
      else:
        raise Exception(f"Error {response.status_code}: {response_body['message']}")
    # Dynamic dict parsing
    api_vitals = response_body.get("vital_signs", {})
    sig_ds = {}
    conf_ds = {}
    for name, obj in api_vitals.items():
      if 'data' in obj:
        sig_ds[name] = np.asarray(obj['data'])
        c_val = obj.get('confidence')
        if c_val is not None:
          conf_ds[name] = np.asarray(c_val)
        else:
          conf_ds[name] = np.zeros_like(sig_ds[name])
    live_ds = np.asarray(response_body["face"]["confidence"])
    if self.op_mode == Mode.BURST:
      self.state = np.asarray(response_body["state"]["data"], dtype=np.float32)
    return sig_ds, conf_ds, live_ds, idxs

  def postprocess(
      self,
      sig: np.ndarray,
      fps: float,
      type: str = 'ppg_waveform',
      filter: bool = True
    ) -> np.ndarray:
    """Apply filters to the estimated signal. 
    Args:
      sig: The estimated signal. Shape (n_frames,)
      fps: The rate at which video was sampled. Scalar
      type: The signal type - either 'ppg_waveform' or 'respiratory_waveform'.
      filter: Whether to apply filters
    Returns:
      x: The filtered signal. Shape (n_frames,)
    """
    n_frames = sig.shape[-1]
    if filter:
      # Get filter parameters
      if type == 'ppg_waveform':
        size = moving_average_size_for_hr_response(fps)
        Lambda = detrend_lambda_for_hr_response(fps)
      elif type == 'respiratory_waveform':
        size = moving_average_size_for_rr_response(fps)
        Lambda = detrend_lambda_for_rr_response(fps)
      else:
        raise ValueError(f"Type {type} not implemented!")
      if sig.shape[-1] < 4 * API_MAX_FRAMES:
        # Detrend only for shorter videos for performance reasons
        sig = detrend(sig, Lambda)
      sig = moving_average(sig, size)
      sig = standardize(sig)
    assert sig.shape == (n_frames,)
    return sig

  def reset(self):
    """Reset"""
    if self.op_mode == Mode.BURST:
      self.state = None
      self.input_buffer = None

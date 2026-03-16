# Copyright (c) 2026 Rouast Labs
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
import gzip
import math
import numpy as np
from prpy.numpy.face import get_roi_from_det
from prpy.numpy.image import probe_image_inputs, parse_image_inputs
from prpy.numpy.interp import interpolate_filtered
from prpy.numpy.utils import enough_memory_for_ndarray
import json
import logging
import requests
import os
from typing import Union, Tuple
import vitallens_core as vc

from vitallens.constants import API_MAX_FRAMES, API_OVERLAP
from vitallens.constants import API_FILE_URL, API_STREAM_URL, API_RESOLVE_URL
from vitallens.errors import VitalLensAPIKeyError, VitalLensAPIQuotaExceededError, VitalLensAPIError
from vitallens.methods.rppg_method import RPPGMethod
from vitallens.signal import reassemble_from_windows
from vitallens.utils import check_faces_in_roi

def _resolve_model_config(
    api_key: str,
    model_name: str,
    proxies: dict = None
  ) -> vc.SessionConfig:
  """Calls the /resolve-model endpoint to get the correct model config.
  
  Args:
    api_key: The API key
    model_name: The requested model name (e.g., 'vitallens', 'vitallens-2.0')
    proxies: Dictionary mapping protocol to the URL of the proxy
  Returns:
    resolved_config: The SessionConfig of the resolved model 
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
    config_dict = data['config']
    return vc.SessionConfig(
      model_name=data['resolved_model'],
      supported_vitals=config_dict.get('supported_vitals', []),
      fps_target=float(config_dict.get('fps_target', 30.0)),
      input_size=int(config_dict.get('input_size', 40)),
      n_inputs=int(config_dict.get('n_inputs', 4)),
      roi_method=config_dict.get('roi_method', 'upper_body_cropped'),
      return_waveforms=['ppg_waveform', 'respiratory_waveform']
    )
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
      api_key: str,
      requested_model_name: str,
      proxies: dict = None
    ):
    """Initialize the `VitalLensRPPGMethod`
    
    Args:
      api_key: The API key
      requested_model_name: The requested VitalLens model name,
      proxies: Dictionary mapping protocol to the URL of the proxy
    """
    super(VitalLensRPPGMethod, self).__init__()
    if proxies is None and (api_key is None or api_key == ''):
      raise VitalLensAPIKeyError()
    self.api_key = api_key
    self.proxies = proxies
    self.session_config = _resolve_model_config(api_key=api_key, model_name=requested_model_name, proxies=proxies)
    self.parse_config(self.session_config)
    self.resolved_model = self.session_config.model_name
    self.requested_model_name = requested_model_name if requested_model_name != "vitallens" else None
    self.http_session = requests.Session()

  def parse_config(self, config: vc.SessionConfig):
    """Set properties based on the config.
    
    Args:
      config: The method's SessionConfig
    """
    super(VitalLensRPPGMethod, self).parse_config(config=config)
    self.n_inputs = config.n_inputs
    self.input_size = config.input_size
    vitals = [vc.get_vital_info(v).id for v in config.supported_vitals if vc.get_vital_info(v) is not None]
    waveforms = config.return_waveforms or []
    self.signals = set(vitals + waveforms)
    self.est_window_length = 0
    self.est_window_overlap = 0

  def infer_batch(
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
       - sig_dict: A dictionary of the estimated signals.
       - conf_dict: A dictionary of the estimated confidences.
       - live: The face live confidence. Shape (n_frames,)
    """
    inputs_shape, fps, video_issues = probe_image_inputs(inputs, fps=fps)
    fps_target = override_fps_target if override_fps_target is not None else self.fps_target
    inputs_n = inputs_shape[0]
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
      sig_dict = dict(zip(keys, sig))
      conf_dict = dict(zip(keys, conf))
    else:
      sig_dict, conf_dict = {}, {}
    # Aggregate liveness
    live_stacked = np.array(live_results)[:, np.newaxis, :]
    rec_live, _ = reassemble_from_windows(x=live_stacked, idxs=idxs_results)
    live = interpolate_filtered(t_in=idxs, s_in=rec_live[0], t_out=np.arange(inputs_n), axis=0, extrapolate=True)
    # Assemble and return the results
    return sig_dict, conf_dict, live

  def infer_stream(self, frames: np.ndarray, fps: float, state=None):
    """Estimate vitals from a sequence of frames using the VitalLens streaming API.

    Args:
      frames: The input video frames of shape (n_frames, h, w, 3).
      fps: The sampling frequency of the input frames.
      state: The internal state of the rPPG method used to maintain temporal continuity.
    Returns:
      Tuple of
        - sig_dict: A dictionary of the estimated signals.
        - conf_dict: A dictionary of the estimated confidences.
        - live: The face live confidence. Shape (n_frames,)
        - new_state: The updated internal state of the rPPG method.
    """
    headers = {
      "Content-Type": "application/octet-stream",
      "X-Encoding": "gzip"
    }
    if self.api_key:
      headers["x-api-key"] = self.api_key
    origin = os.getenv('VITALLENS_API_ORIGIN', 'vitallens-python')
    headers["X-Origin"] = origin
    if self.requested_model_name:
      headers["X-Model"] = self.requested_model_name
    if state is not None:
      state_bytes = np.asarray(state, dtype=np.float32).tobytes()
      headers["X-State"] = base64.b64encode(state_bytes).decode('utf-8')

    # Compress the raw video bytes
    raw_rgb_bytes = frames.tobytes()
    compressed_data = gzip.compress(raw_rgb_bytes)

    # Post the binary payload
    response = self.http_session.post(API_STREAM_URL, headers=headers, data=compressed_data, proxies=self.proxies)

    if response.status_code != 200:
      response_body = response.json()
      logging.error(f"Error {response.status_code}: {response_body.get('message')}")
      if response.status_code == 403: raise VitalLensAPIKeyError()
      elif response.status_code == 429: raise VitalLensAPIQuotaExceededError()
      else: raise VitalLensAPIError(f"API Error: {response_body.get('message')}")

    response_body = response.json()

    api_waveforms = response_body.get("waveforms", {})
    sig_dict, conf_dict = {}, {}
    for name, obj in api_waveforms.items():
      if 'data' in obj:
        sig_dict[name] = np.asarray(obj['data'])
        conf_dict[name] = np.asarray(obj.get('confidence', [1.0] * len(sig_dict[name])))

    n_res = len(next(iter(sig_dict.values()))) if sig_dict else frames.shape[0]
    live = np.asarray(response_body.get("face", {}).get("confidence", [1.0] * n_res))
    new_state = response_body.get("state", {}).get("data")

    return sig_dict, conf_dict, live, new_state

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
      # Inputs have not been parsed globally. Parse the inputs
      frames_ds, _, _, ds_factor, idxs = parse_image_inputs(
        inputs=inputs, fps=fps, roi=roi, target_size=self.input_size, target_fps=fps_target,
        preserve_aspect_ratio=False, library='prpy', scale_algorithm='bilinear', 
        trim=(start, end) if start is not None and end is not None else None,
        allow_image=False, videodims=True)
    # Make sure we have the correct number of frames
    idxs = np.asarray(idxs)
    expected_n = math.ceil(((end-start) if start is not None and end is not None else inputs_shape[0]) / ds_factor)
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
    # Ask API to process video
    response = self.http_session.post(API_FILE_URL, headers=headers, json=payload, proxies=self.proxies)
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
    api_waveforms = response_body.get("waveforms", {})
    sig_ds = {}
    conf_ds = {}
    for name, obj in api_waveforms.items():
      if 'data' in obj:
        sig_ds[name] = np.asarray(obj['data'])
        c_val = obj.get('confidence')
        if c_val is not None:
          conf_ds[name] = np.asarray(c_val)
        else:
          conf_ds[name] = np.zeros_like(sig_ds[name])
    live_ds = np.asarray(response_body.get("face", {}).get("confidence", [1.0] * frames_ds.shape[0]))
    return sig_ds, conf_ds, live_ds, idxs

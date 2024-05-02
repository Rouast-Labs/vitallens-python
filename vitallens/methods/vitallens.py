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
import numpy as np
from prpy.numpy.face import get_roi_from_det
from prpy.numpy.signal import detrend, moving_average, standardize
from prpy.numpy.signal import interpolate_cubic_spline
import json
import logging
import requests
from typing import Union, Tuple

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
    self.input_size = config['input_size']
    self.roi_method = config['roi_method']
    self.api_key = api_key
  def __call__(
      self,
      frames: Union[np.ndarray, str],
      faces: np.ndarray,
      fps: float,
      override_fps_target: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate vitals from video frames using the VitalLens API.

    Args:
      frames: The video frames. Shape (n_frames, h, w, c)
      faces: The face detection boxes. Shape (n_frames, 4) in form (x0, y0, x1, y1)
      fps: The rate at which video was sampled.
      override_fps_target: Override the method's default inference fps (optional).
    Returns:
      sig: Estimated pulse signal. Shape (1, n_frames)
      conf: Dummy estimation confidence (set to always 1). Shape (1, n_frames)
      live: Dummy liveness estimation (set to always 1). Shape (1, n_frames)
    """
    inputs_shape, fps = probe_video_inputs(video=frames, fps=fps)
    # TODO: Choose face box that face stays most centered in
    # TODO: Warn if face moves too much (more than 1/3 out of chosen face box)
    roi = get_roi_from_det(
      faces[0], roi_method=self.roi_method, clip_dims=(inputs_shape[2], inputs_shape[1]))
    # Parse the inputs
    frames_ds, fps, inputs_shape, ds_factor = parse_video_inputs(
      video=frames, fps=fps, target_size=self.input_size, roi=roi,
      target_fps=override_fps_target if override_fps_target is not None else self.fps_target,
      library='prpy', scale_algorithm='bilinear')
    # Communicate with API
    headers = {"x-api-key": self.api_key}
    payload = {"video": base64.b64encode(frames_ds.tobytes()).decode('utf-8')}
    if frames_ds.shape[0] > 900:
      # TODO: Deal with this
      logging.warn("Too long!")
    # Ask API to process video
    response = requests.post("https://uimunafoxe.execute-api.ap-southeast-2.amazonaws.com/beta/process", headers=headers, json=payload)
    # Check if response was successful
    if response.status_code != 200:
      # TODO: Deal with error
      logging.error("Error {} ({})".format(response.status_code, response.text))
      return [], [], []
    # Parse response
    response_body = json.loads(response.text)
    sig_ds = np.asarray(response_body["signal"])
    conf_ds = np.asarray(response_body["conf"])
    live_ds = np.asarray(response_body["live"])
    # Interpolate to original sampling rate (n_frames,)
    sig = interpolate_cubic_spline(
      x=np.arange(inputs_shape[0])[0::ds_factor], y=sig_ds, xs=np.arange(inputs_shape[0]), axis=1)
    conf = interpolate_cubic_spline(
      x=np.arange(inputs_shape[0])[0::ds_factor], y=conf_ds, xs=np.arange(inputs_shape[0]), axis=1)
    live = interpolate_cubic_spline(
      x=np.arange(inputs_shape[0])[0::ds_factor], y=live_ds, xs=np.arange(inputs_shape[0]), axis=0)
    # Filter (n_frames,)
    sig = [self.postprocess(p, fps, type=name) for p, name in zip(sig, ['pulse', 'resp'])]
    return sig, conf, live
  def postprocess(self, sig, fps, type='pulse', filter=True):
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
      if type == 'pulse':
        size = moving_average_size_for_hr_response(fps)
        Lambda = detrend_lambda_for_hr_response(fps)
      elif type == 'resp':
        size = moving_average_size_for_rr_response(fps)
        Lambda = detrend_lambda_for_rr_response(fps)
      else:
        raise ValueError("Type {} not implemented!".format(type))
      # Detrend
      sig = detrend(sig, Lambda)
      # Moving average
      sig = moving_average(sig, size)
      # Standardize
      sig = standardize(sig)
    # Return
    assert sig.shape == (n_frames,)
    return sig
  
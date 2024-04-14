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
from prpy.numpy.face import get_roi_from_det
from typing import Union, Tuple

from vitallens.methods.rppg_method import RPPGMethod
from vitallens.utils import probe_video_inputs, parse_video_inputs

class VitalLensRPPGMethod(RPPGMethod):
  def __init__(
      self,
      config: dict
    ):
    super(VitalLensRPPGMethod, self).__init__(config=config)
    self.input_size = config['input_size']
    self.roi_method = config['roi_method']
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
    fps_ds = fps*1.0/ds_factor
    n_ds = frames_ds.shape[0]
    
    # TODO: Implement communication with server
    
    return np.zeros(n_ds), np.zeros(n_ds), np.zeros(n_ds)

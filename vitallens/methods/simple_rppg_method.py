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

import abc
import numpy as np
from prpy.numpy.face import get_roi_from_det
from prpy.numpy.signal import interpolate_cubic_spline
from typing import Union, Tuple

from vitallens.methods.rppg_method import RPPGMethod
from vitallens.utils import parse_video_inputs, merge_faces

class SimpleRPPGMethod(RPPGMethod):
  def __init__(
      self,
      config: dict
    ):
    super(SimpleRPPGMethod, self).__init__(config=config)
    self.roi_method = config['roi_method']
  @abc.abstractmethod
  def algorithm(
      self,
      rgb: np.ndarray,
      fps: float
    ):
    pass
  @abc.abstractmethod
  def pulse_filter(self, 
      sig: np.ndarray,
      fps: float
    ) -> np.ndarray:
    pass
  def __call__(
      self,
      frames: Union[np.ndarray, str],
      faces: np.ndarray,
      fps: float,
      override_fps_target: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate pulse signal from video frames using the subclass algorithm.

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
    # Compute temporal union of ROIs
    u_roi = merge_faces(faces)
    faces = faces - [u_roi[0], u_roi[1], u_roi[0], u_roi[1]]
    # Parse the inputs
    frames_ds, fps, inputs_shape, ds_factor = parse_video_inputs(
      video=frames, fps=fps, target_size=None, roi=u_roi,
      target_fps=override_fps_target if override_fps_target is not None else self.fps_target)
    assert inputs_shape[0] == faces.shape[0], "Need same number of frames as face detections"
    faces_ds = faces[0::ds_factor]
    assert frames_ds.shape[0] == faces_ds.shape[0], "Need same number of frames as face detections"
    fps_ds = fps*1.0/ds_factor
    n_ds = frames_ds.shape[0]
    # Extract rgb signal (n_frames_ds, 3)
    roi_ds = np.asarray([get_roi_from_det(f, roi_method=self.roi_method) for f in faces_ds]) # roi for each frame (n, 4)
    rgb_ds = np.asarray([np.mean(frames_ds[i, roi_ds[i,1]:roi_ds[i,3], roi_ds[i,0]:roi_ds[i,2]], axis=(0,1)) for i in range(n_ds)])
    # Perform rppg algorithm step (n_frames_ds,)
    sig_ds = self.algorithm(rgb_ds, fps_ds)
    # Interpolate to original sampling rate (n_frames,)
    sig = interpolate_cubic_spline(
      x=np.arange(inputs_shape[0])[0::ds_factor], y=sig_ds, xs=np.arange(inputs_shape[0]), axis=1)
    # Filter (n_frames,)
    sig = self.pulse_filter(sig, fps)
    # Add conf and live (n_frames,)
    conf = np.ones_like(sig)
    live = np.ones_like(sig)
    # Return (1, n_frames)
    return sig[np.newaxis], conf[np.newaxis], live[np.newaxis]

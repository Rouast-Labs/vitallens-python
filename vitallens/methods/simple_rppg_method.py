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

import abc
import numpy as np
from prpy.numpy.face import get_roi_from_det
from prpy.numpy.image import reduce_roi, parse_image_inputs
from prpy.numpy.interp import interpolate_filtered
from typing import Union, Tuple
import vitallens_core as vc

from vitallens.methods.rppg_method import RPPGMethod
from vitallens.utils import merge_faces

class SimpleRPPGMethod(RPPGMethod):
  """A simple rPPG method using a handcrafted algorithm based on RGB signal trace"""
  def __init__(self):
    """Initialize the `SimpleRPPGMethod`"""
    super(SimpleRPPGMethod, self).__init__()
    self.n_inputs = 1

  def parse_config(
      self,
      config: vc.SessionConfig,
      est_window_length: int,
      est_window_overlap: int
    ):
    """Set properties based on the config.
    
    Args:
      config: The method's config object
      est_window_length: The length of the estimation window
      est_window_overlap: The overlap of consecutive estimation windows
    """
    super(SimpleRPPGMethod, self).parse_config(config=config)
    self.signals = config.supported_vitals + (config.return_waveforms or [])
    self.est_window_length = est_window_length
    self.est_window_overlap = est_window_overlap
    self.est_window_flexible = self.est_window_length == 0

  @abc.abstractmethod
  def algorithm(self, rgb: np.ndarray, fps: float):
    """The algorithm. Abstract method to be implemented by subclasses."""
    pass

  def infer_batch(
      self,
      inputs: Union[np.ndarray, str],
      faces: np.ndarray,
      fps: float,
      override_fps_target: float = None,
      override_global_parse: bool = None,
    ) -> Tuple[dict, dict, dict, dict, np.ndarray]:
    """Estimate pulse signal from video frames using the subclass algorithm.

    Args:
      inputs: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        in unscaled uint8 RGB format, or a path to a video file.
      faces: The face detection boxes as np.int64. Shape (n_frames, 4) in form (x0, y0, x1, y1)
      fps: The rate at which video was sampled.
      override_fps_target: Override the method's default inference fps (optional).
      override_global_parse: Has no effect here.
    Returns:
      Tuple of
        - data: A dictionary with the values of the estimated vital signs.
        - unit: A dictionary with the units of the estimated vital signs.
        - conf: A dictionary with the confidences of the estimated vital signs.
        - note: A dictionary with notes on the estimated vital signs.
        - live: Dummy live confidence estimation (set to always 1). Shape (n_frames,)
    """
    # Compute temporal union of ROIs
    u_roi = merge_faces(faces)
    faces = faces - [u_roi[0], u_roi[1], u_roi[0], u_roi[1]]
    # Parse the inputs
    target_fps = override_fps_target if override_fps_target is not None else self.fps_target
    frames_ds, fps, inputs_shape, ds_factor, _ = parse_image_inputs(
      inputs=inputs, fps=fps, roi=u_roi, target_size=None, target_fps=target_fps,
      preserve_aspect_ratio=False, library='prpy', scale_algorithm='bilinear', 
      trim=None, allow_image=False, videodims=True)
    assert inputs_shape[0] == faces.shape[0], "Need same number of frames as face detections"
    faces_ds = faces[0::ds_factor]
    assert frames_ds.shape[0] == faces_ds.shape[0], "Need same number of frames as face detections"
    fps_ds = fps*1.0/ds_factor
    # Extract rgb signal (n_frames_ds, 3)
    roi_ds = np.asarray([get_roi_from_det(f, roi_method=self.roi_method) for f in faces_ds], dtype=np.int64) # roi for each frame (n, 4)
    rgb_ds = reduce_roi(video=frames_ds, roi=roi_ds)
    # Perform rppg algorithm step (n_frames_ds,)
    sig_ds = self.algorithm(rgb_ds, fps_ds)
    # Interpolate to original sampling rate (n_frames,)
    sig = interpolate_filtered(t_in=np.arange(inputs_shape[0])[0::ds_factor],
                               s_in=sig_ds,
                               t_out=np.arange(inputs_shape[0]),
                               axis=0, extrapolate=True)
    # Filter (n_frames,)
    # sig = self.pulse_filter(sig, fps)
    # Package into dict
    sig_dict = {'ppg_waveform': sig}
    conf_dict = {'ppg_waveform': np.ones_like(sig)}
    live = np.ones_like(sig)
    # Assemble and return the results
    return sig_dict, conf_dict, live

  def infer_stream(
      self,
      frames: np.ndarray,
      fps: float,
      state
    ):
    """Estimate pulse signal from a sequence of frames in a streaming context.

    Args:
      frames: The input video frames of shape (n_frames, h, w, 3).
      fps: The sampling frequency of the input frames.
      state: The internal state of the rPPG method (unused for simple methods).
    Returns:
      Tuple of
        - sig_dict: A dictionary of the estimated signals.
        - conf_dict: A dictionary of the estimated confidences.
        - live_out: Dummy live confidence estimation (set to always 1). Shape (n_frames,)
        - state: The updated internal state of the rPPG method (None).
    """
    sig = self.algorithm(frames, fps)
    sig_dict = {'ppg_waveform': sig}
    conf_dict = {'ppg_waveform': np.ones_like(sig)}
    live_out = np.ones_like(sig)
    return sig_dict, conf_dict, live_out, None

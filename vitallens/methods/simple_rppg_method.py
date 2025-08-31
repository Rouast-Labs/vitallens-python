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
from prpy.numpy.image import reduce_roi, parse_image_inputs
from prpy.numpy.interp import interpolate_filtered
from prpy.numpy.physio import CALC_HR_MIN_T
from typing import Union, Tuple

from vitallens.buffer import SignalBuffer
from vitallens.enums import Mode
from vitallens.methods.rppg_method import RPPGMethod
from vitallens.signal import assemble_results
from vitallens.utils import merge_faces

class SimpleRPPGMethod(RPPGMethod):
  """A simple rPPG method using a handcrafted algorithm based on RGB signal trace"""
  def __init__(
      self,
      mode: Mode
    ):
    """Initialize the `SimpleRPPGMethod`
    
    Args:
      config: The configuration dict
      mode: The operation mode
    """
    super(SimpleRPPGMethod, self).__init__(mode=mode)
    self.n_inputs = 1
  def parse_config(
      self,
      config: dict
    ):
    """Set properties based on the config.
    
    Args:
      config: The method's config dict
    """
    super(SimpleRPPGMethod, self).parse_config(config=config)
    self.signals = config['signals']
    if self.op_mode == Mode.BURST:
      self.buffer = SignalBuffer(size=self.est_window_length, ndim=2)
  @abc.abstractmethod
  def algorithm(
      self,
      rgb: np.ndarray,
      fps: float
    ):
    """The algorithm. Abstract method to be implemented by subclasses."""
    pass
  @abc.abstractmethod
  def pulse_filter(self, 
      sig: np.ndarray,
      fps: float
    ) -> np.ndarray:
    """The post-processing filter to be applied to estimated pulse signal. Abstract method to be implemented by subclasses."""
    pass
  def __call__(
      self,
      inputs: Union[np.ndarray, str],
      faces: np.ndarray,
      fps: float,
      override_fps_target: float = None,
      override_global_parse: float = None,
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
    frames_ds, fps, inputs_shape, ds_factor, _ = parse_image_inputs(
      inputs=inputs, fps=fps, roi=u_roi, target_size=None,
      target_fps=override_fps_target if override_fps_target is not None else self.fps_target,
      preserve_aspect_ratio=False, library='prpy', scale_algorithm='bilinear', 
      trim=None, allow_image=False, videodims=True)
    assert inputs_shape[0] == faces.shape[0], "Need same number of frames as face detections"
    faces_ds = faces[0::ds_factor]
    assert frames_ds.shape[0] == faces_ds.shape[0], "Need same number of frames as face detections"
    fps_ds = fps*1.0/ds_factor
    # Extract rgb signal (n_frames_ds, 3)
    if self.op_mode == Mode.BATCH:
      roi_ds = np.asarray([get_roi_from_det(f, roi_method=self.roi_method) for f in faces_ds], dtype=np.int64) # roi for each frame (n, 4)
      rgb_ds = reduce_roi(video=frames_ds, roi=roi_ds)
    else:
      # Use the last face detection for cropping (n_frames, 3)
      rgb_ds = reduce_roi(video=frames_ds, roi=np.asarray(get_roi_from_det(faces_ds[-1], roi_method=self.roi_method), dtype=np.int64))
      # Push to buffer and get buffered vals (pred_window_length, 3)
      rgb_ds = self.buffer.update(rgb_ds, dt=inputs.shape[0])
    # Perform rppg algorithm step (n_frames_ds,)
    sig_ds = self.algorithm(rgb_ds, fps_ds)
    # Interpolate to original sampling rate (n_frames,)
    sig = interpolate_filtered(t_in=np.arange(inputs_shape[0])[0::ds_factor],
                               s_in=sig_ds,
                               t_out=np.arange(inputs_shape[0]),
                               axis=1, extrapolate=True)
    # Filter and add dim (1, n_frames)
    sig = self.pulse_filter(sig, fps)
    sig = np.expand_dims(sig, axis=0)
    # Simple rPPG method cannot specify confidence or live. Set to always 1.
    conf = np.ones_like(sig)
    live = np.ones_like(sig[0])
    # Assemble and return the results
    return assemble_results(sig=sig,
                            conf=conf,
                            live=live,
                            fps=fps,
                            train_sig_names=['ppg_waveform'],
                            pred_signals=self.signals,
                            method=self.method,
                            min_t_hr=CALC_HR_MIN_T,
                            can_provide_confidence=False)
  def reset(self):
    """Reset"""
    if self.op_mode == Mode.BURST:
      self.buffer.clear()

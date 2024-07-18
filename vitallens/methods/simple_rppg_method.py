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
from prpy.numpy.image import reduce_roi
from prpy.numpy.signal import interpolate_cubic_spline, estimate_freq
from typing import Union, Tuple

from vitallens.constants import SECONDS_PER_MINUTE, CALC_HR_MIN, CALC_HR_MAX
from vitallens.methods.rppg_method import RPPGMethod
from vitallens.utils import parse_video_inputs, merge_faces

class SimpleRPPGMethod(RPPGMethod):
  def __init__(
      self,
      config: dict
    ):
    super(SimpleRPPGMethod, self).__init__(config=config)
    self.model = config['model']
    self.roi_method = config['roi_method']
    self.signals = config['signals']
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
    ) -> Tuple[dict, dict, dict, dict, np.ndarray]:
    """Estimate pulse signal from video frames using the subclass algorithm.

    Args:
      frames: The video frames. Shape (n_frames, h, w, c)
      faces: The face detection boxes as np.int64. Shape (n_frames, 4) in form (x0, y0, x1, y1)
      fps: The rate at which video was sampled.
      override_fps_target: Override the method's default inference fps (optional).
    Returns:
      data: A dictionary with the values of the estimated vital signs.
      unit: A dictionary with the units of the estimated vital signs.
      conf: A dictionary with the confidences of the estimated vital signs.
      note: A dictionary with notes on the estimated vital signs.
      live: Dummy live confidence estimation (set to always 1). Shape (1, n_frames)
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
    # Extract rgb signal (n_frames_ds, 3)
    roi_ds = np.asarray([get_roi_from_det(f, roi_method=self.roi_method) for f in faces_ds], dtype=np.int64) # roi for each frame (n, 4)
    rgb_ds = reduce_roi(video=frames_ds, roi=roi_ds)
    # Perform rppg algorithm step (n_frames_ds,)
    sig_ds = self.algorithm(rgb_ds, fps_ds)
    # Interpolate to original sampling rate (n_frames,)
    sig = interpolate_cubic_spline(
      x=np.arange(inputs_shape[0])[0::ds_factor], y=sig_ds, xs=np.arange(inputs_shape[0]), axis=1)
    # Filter (n_frames,)
    sig = self.pulse_filter(sig, fps)
    # Estimate HR
    hr = estimate_freq(
      sig, f_s=fps, f_res=0.1/SECONDS_PER_MINUTE,
      f_range=(CALC_HR_MIN/SECONDS_PER_MINUTE, CALC_HR_MAX/SECONDS_PER_MINUTE),
      method='periodogram') * SECONDS_PER_MINUTE
    # Assemble results
    data, unit, conf, note = {}, {}, {}, {}
    for name in self.signals:
      if name == 'heart_rate':
        data[name] = hr
        unit[name] = 'bpm'
        conf[name] = 1.0
        note[name] = 'Estimate of the heart rate using {} method. This method is not capable of providing a confidence estimate, hence returning 1.'.format(self.model)
      elif name == 'ppg_waveform':
        data[name] = sig
        unit[name] = 'unitless'
        conf[name] = np.ones_like(sig)
        note[name] = 'Estimate of the ppg waveform using {} method. This method is not capable of providing a confidence estimate, hence returning 1.'.format(self.model)
    # Return results
    return data, unit, conf, note, np.ones_like(sig)

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
from typing import Union

from .ssd import FaceDetector
from .utils import probe_inputs, parse_inputs, check_faces

logging.getLogger().setLevel("INFO")

class VitalLens:
  def __init__(
      self, 
      detect_faces: bool = True,
      fdet_max_faces: int = 2,
      fdet_fs: float = 1.0,
      fdet_score_threshold: float = 0.9,
      fdet_iou_threshold: float = 0.3
    ):
    """Initialisation. Loads face detection model if necessary.

    Args:
      detect_faces: `True` if faces need to be detected. If `False`, VitalLens
        will assume frames have been cropped to a stable ROI with a single face.
      fdet_max_faces: The maximum number of faces to detect (if necessary).
      fdet_fs: Frequency [Hz] at which faces should be scanned. Detections are
        linearly interpolated for remaining frames.
      fdet_score_threshold: Face detection score threshold.
      fdet_iou_threshold: Face detection iou threshold.
    """
    # TODO: Pass API key
    self.detect_faces = detect_faces
    self.fdet_max_faces = fdet_max_faces
    self.fdet_fs = fdet_fs
    self.fdet_score_threshold = fdet_score_threshold
    self.fdet_iou_threshold = fdet_iou_threshold
    if detect_faces:
      # Initialise face detector
      self.face_detector = FaceDetector(
        max_faces=fdet_max_faces, fs=fdet_fs, score_threshold=fdet_score_threshold,
        iou_threshold=fdet_iou_threshold)
  def __call__(
      self,
      inputs: Union[np.ndarray, str],
      faces: Union[np.ndarray, list] = None,
      fps: float = None,
      rppg_fps_target: float = 30.0
    ) -> None:
    """Run rPPG inference.

    Args:
      inputs: The inputs to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        with a sequence of frames in unscaled uint8 RGB format, or a path to a
        video file. Note that aggressive video encoding destroys the rPPG signal.
      faces: Face boxes in flat point form, containing [x0, y0, x1, y1] coords. Required
        if detect_faces=False, otherwise ignored. list or np.ndarray of
        shape (n_faces, n_frames, 4) for multiple faces detected on multiple frames,
        shape (n_frames, 4) for single face detected on mulitple frames, or
        shape (4,) for a single face detected globally, or
        `None` to assume all frames already cropped to the same single face detection.
      fps: Sampling frequency of the input video.
        Required if type(inputs) == np.ndarray. 
      rppg_fps_target: Target fps at which rPPG inference should be run.
    Returns:
      result: TODO
    """
    # Probe inputs
    inputs_shape, fps = probe_inputs(inputs=inputs, fps=fps)
    n_frames, height, width, _ = inputs_shape
    if self.detect_faces:
      # Detect faces
      faces_rel, _ = self.face_detector(inputs=inputs, fps=fps)
      # If no faces detected: return empty list
      if len(faces_rel) == 0:
        logging.warn("No faces to analyze")
        return []
      # Convert to absolute units
      faces = (faces_rel * [width, height, width, height]).astype(int)
      # Face axis first
      faces = np.transpose(faces, (1, 0, 2))
    # Check if the faces are valid
    faces = check_faces(faces, inputs_shape)  
    print("faces: {}".format(faces))
    
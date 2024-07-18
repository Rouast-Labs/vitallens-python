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

from enum import IntEnum
import logging
import numpy as np
from typing import Union

from vitallens.constants import DISCLAIMER
from vitallens.methods.g import GRPPGMethod
from vitallens.methods.chrom import CHROMRPPGMethod
from vitallens.methods.pos import POSRPPGMethod
from vitallens.methods.vitallens import VitalLensRPPGMethod
from vitallens.ssd import FaceDetector
from vitallens.utils import load_config, probe_video_inputs, check_faces

class Method(IntEnum):
  VITALLENS = 1
  G = 2
  CHROM = 3
  POS = 4

logging.getLogger().setLevel("INFO")

class VitalLens:
  def __init__(
      self, 
      method: Method = Method.VITALLENS,
      api_key: str = None,
      detect_faces: bool = True,
      fdet_max_faces: int = 2,
      fdet_fs: float = 1.0,
      fdet_score_threshold: float = 0.9,
      fdet_iou_threshold: float = 0.3
    ):
    """Initialisation. Loads face detection model if necessary.

    Args:
      method: The rPPG method to be used for inference.
      api_key: Usage key for the VitalLens API (required for Method.VITALLENS)
      detect_faces: `True` if faces need to be detected, otherwise `False`.
      fdet_max_faces: The maximum number of faces to detect (if necessary).
      fdet_fs: Frequency [Hz] at which faces should be scanned. Detections are
        linearly interpolated for remaining frames.
      fdet_score_threshold: Face detection score threshold.
      fdet_iou_threshold: Face detection iou threshold.
    """
    self.api_key = api_key
    # Load the config and model
    self.config = load_config(method.name.lower() + ".yaml")
    if self.config['model'] == 'g':
      self.rppg = GRPPGMethod(self.config)
    elif self.config['model'] == 'chrom':
      self.rppg = CHROMRPPGMethod(self.config)
    elif self.config['model'] == 'pos':
      self.rppg = POSRPPGMethod(self.config)
    elif self.config['model'] == 'vitallens':
      if self.api_key is None or self.api_key == '':
        raise ValueError("An API key is required to use Method.VITALLENS, but was not provided. "
                         "Get one for free at https://www.rouast.com/api.")
      self.rppg = VitalLensRPPGMethod(self.config, self.api_key)
    else:
      raise ValueError("Method {} not implemented!".format(self.config['model']))
    self.detect_faces = detect_faces
    if detect_faces:
      # Initialise face detector
      self.face_detector = FaceDetector(
        max_faces=fdet_max_faces, fs=fdet_fs, score_threshold=fdet_score_threshold,
        iou_threshold=fdet_iou_threshold)
  def __call__(
      self,
      video: Union[np.ndarray, str],
      faces: Union[np.ndarray, list] = None,
      fps: float = None,
      override_fps_target: float = None
    ) -> list:
    """Run rPPG inference.

    Args:
      video: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        with a sequence of frames in unscaled uint8 RGB format, or a path to a
        video file. Note that aggressive video encoding destroys the rPPG signal.
      faces: Face boxes in flat point form, containing [x0, y0, x1, y1] coords.
        Ignored unless detect_faces=False. Pass a list or np.ndarray of
        shape (n_faces, n_frames, 4) for multiple faces detected on multiple frames,
        shape (n_frames, 4) for single face detected on mulitple frames, or
        shape (4,) for a single face detected globally, or
        `None` to assume all frames already cropped to the same single face detection.
      fps: Sampling frequency of the input video. Required if type(video) == np.ndarray. 
      override_fps_target: Target fps at which rPPG inference should be run (optional).
        If not provided, will use default of the selected method.
    Returns:
      result: Analysis results as a list of faces in the following format:
        [
          {
            'face': {
              'coordinates': <Face coordinates for each frame as np.ndarray of shape (n_frames, 4)>,
              'confidence': <Face live confidence for each frame as np.ndarray of shape (n_frames,)>,
              'note': <Explanatory note>
            },
            'vital_signs': {
              'heart_rate': {
                  'value': <Estimated value as float scalar>,
                  'unit': <Value unit>,
                  'confidence': <Estimation confidence as float scalar>,
                  'note': <Explanatory note>
                },
              'respiratory_rate': {
                'value': <Estimated value as float scalar>,
                'unit': <Value unit>,
                'confidence': <Estimation confidence as float scalar>,
                'note': <Explanatory note>
              },
              'ppg_waveform': {
                'data': <Estimated waveform value for each frame as np.ndarray of shape (n_frames,)>,
                'unit': <Data unit>,
                'confidence': <Estimation confidence for each frame as np.ndarray of shape (n_frames,)>,
                'note': <Explanatory note>
              },
              'respiratory_waveform': {
                'data': <Estimated waveform value for each frame as np.ndarray of shape (n_frames,)>,
                'unit': <Data unit>,
                'confidence': <Estimation confidence for each frame as np.ndarray of shape (n_frames,)>,
                'note': <Explanatory note>
              },
            },
            "message": <Message about estimates>
          },
          { 
            <same structure for face 2 if present>
          },
          ...
        ]
    """
    # Probe inputs
    inputs_shape, fps = probe_video_inputs(video=video, fps=fps)
    _, height, width, _ = inputs_shape
    if self.detect_faces:
      # Detect faces
      faces_rel, _ = self.face_detector(inputs=video, fps=fps)
      # If no faces detected: return empty list
      if len(faces_rel) == 0:
        logging.warn("No faces to analyze")
        return []
      # Convert to absolute units
      faces = (faces_rel * [width, height, width, height]).astype(np.int64)
      # Face axis first
      faces = np.transpose(faces, (1, 0, 2))
    # Check if the faces are valid
    faces = check_faces(faces, inputs_shape)  
    # Run separately for each face
    results = []
    for face in faces:
      # Run selected rPPG method
      data, unit, conf, note, live = self.rppg(
        frames=video, faces=face, fps=fps, override_fps_target=override_fps_target)
      # Parse face results
      face_result = {'face': {
        'coordinates': face,
        'confidence': live,
        'note': "Face detection coordinates for this face, along with live confidence levels."
      }}
      # Parse vital signs results
      vital_signs_results = {}
      for name in self.config['signals']:
        vital_signs_results[name] = {
          '{}'.format('data' if 'waveform' in name else 'value'): data[name],
          'unit': unit[name],
          'confidence': conf[name],
          'note': note[name]
        }
      face_result['vital_signs'] = vital_signs_results
      face_result['message'] = DISCLAIMER
      results.append(face_result)
    return results

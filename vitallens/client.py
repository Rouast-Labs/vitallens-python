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

from datetime import datetime
import json
import logging
import numpy as np
import os
from prpy.numpy.image import probe_image_inputs
from prpy.numpy.physio import EScope, EMethod
from prpy.numpy.physio import estimate_hr_from_signal, estimate_rr_from_signal
from prpy.numpy.physio import CALC_HR_MAX_T, CALC_RR_MAX_T
from prpy.numpy.rolling import rolling_calc
from typing import Union

from vitallens.constants import DISCLAIMER, API_MAX_FRAMES
from vitallens.enums import Method, Mode
from vitallens.methods.g import GRPPGMethod
from vitallens.methods.chrom import CHROMRPPGMethod
from vitallens.methods.pos import POSRPPGMethod
from vitallens.methods.vitallens import VitalLensRPPGMethod
from vitallens.ssd import FaceDetector
from vitallens.utils import load_config, check_faces, convert_ndarray_to_list

logging.getLogger().setLevel("INFO")

class VitalLens:
  def __init__(
      self, 
      method: Method = Method.VITALLENS,
      mode: Mode = Mode.BATCH,
      api_key: str = None,
      detect_faces: bool = True,
      estimate_rolling_vitals: bool = True,
      fdet_max_faces: int = 1,
      fdet_fs: float = 1.0,
      fdet_score_threshold: float = 0.9,
      fdet_iou_threshold: float = 0.3,
      export_to_json: bool = True,
      export_dir: str = "."
    ):
    """Initialisation. Loads face detection model if necessary.

    Args:
      method: The rPPG method to be used for inference.
      mode: Operate in batch or burst mode
      api_key: Usage key for the VitalLens API (required for Method.VITALLENS)
      detect_faces: `True` if faces need to be detected, otherwise `False`.
      estimate_rolling_vitals: Set `True` to compute rolling vitals (e.g., `rolling_heart_rate`).
      fdet_max_faces: The maximum number of faces to detect (if necessary).
      fdet_fs: Frequency [Hz] at which faces should be scanned. Detections are
        linearly interpolated for remaining frames.
      fdet_score_threshold: Face detection score threshold.
      fdet_iou_threshold: Face detection iou threshold.
      export_to_json: If `True`, write results to a json file.
      export_dir: The directory to which json files are written.
    """
    self.api_key = api_key
    # Load the config and model
    self.config = load_config(method.name.lower() + ".yaml")
    self.method = method
    self.mode = mode
    if self.config['model'] == 'g':
      self.rppg = GRPPGMethod(config=self.config, mode=mode)
    elif self.config['model'] == 'chrom':
      self.rppg = CHROMRPPGMethod(config=self.config, mode=mode)
    elif self.config['model'] == 'pos':
      self.rppg = POSRPPGMethod(config=self.config, mode=mode)
    elif self.config['model'] == 'vitallens':
      if self.api_key is None or self.api_key == '':
        raise ValueError("An API key is required to use Method.VITALLENS, but was not provided. "
                         "Get one for free at https://www.rouast.com/api.")
      self.rppg = VitalLensRPPGMethod(config=self.config, mode=mode, api_key=self.api_key)
    else:
      raise ValueError(f"Method {self.config['model']} not implemented!")
    self.detect_faces = detect_faces
    self.estimate_rolling_vitals = estimate_rolling_vitals
    self.export_to_json = export_to_json
    self.export_dir = export_dir
    if detect_faces:
      # Initialise face detector
      self.face_detector = FaceDetector(
        max_faces=fdet_max_faces, fs=fdet_fs, score_threshold=fdet_score_threshold,
        iou_threshold=fdet_iou_threshold)
    assert not (fdet_max_faces > 1 and mode == Mode.BURST), "burst mode only supported for one face"
  def __call__(
      self,
      video: Union[np.ndarray, str],
      faces: Union[np.ndarray, list] = None,
      fps: float = None,
      override_fps_target: float = None,
      override_global_parse: bool = None,
      export_filename: str = None
    ) -> list:
    """Run rPPG inference.

    Args:
      video: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        with a sequence of frames in unscaled uint8 RGB format, or a path to a
        video file. Note that aggressive video encoding destroys the rPPG signal.
      faces: Face boxes in flat point form, containing [x0, y0, x1, y1] coords.
        Ignored unless detect_faces=False. Pass a list or np.ndarray of
        - shape (n_faces, n_frames, 4) for multiple faces detected on multiple frames,
        - shape (n_frames, 4) for single face detected on mulitple frames, or
        - shape (4,) for a single face detected globally, or
        - `None` to assume all frames already cropped to the same single face detection.
      fps: Sampling frequency of the input video. Required if type(video) == np.ndarray. 
      override_fps_target: Target fps at which rPPG inference should be run (optional).
        If not provided, will use default of the selected method.
      override_global_parse: If True, always use global parse. If False, don't use global parse.
        If None, choose based on video.
      export_filename: Filename for json export if applicable.
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
    if self.mode == Mode.BURST and not isinstance(video, np.ndarray):
      raise ValueError("Must provide `np.ndarray` inputs for burst mode.")
    if self.mode == Mode.BURST and video.shape[0] > (API_MAX_FRAMES - self.rppg.n_inputs + 1):
      raise ValueError(f"Maximum number of frames in burst mode is {API_MAX_FRAMES - self.rppg.n_inputs + 1}, but received {video.shape[0]}.")
    inputs_shape, fps, _ = probe_image_inputs(video, fps=fps, allow_image=False)
    # TODO: Optimize performance of simple rPPG methods for long videos
    # Warning if using long video
    target_fps = override_fps_target if override_fps_target is not None else self.rppg.fps_target
    if self.method != Method.VITALLENS and inputs_shape[0]/fps*target_fps > 3600:
      logging.warning("Inference for long videos has yet to be optimized for POS / G / CHROM. This may run out of memory and crash.")
    _, height, width, _ = inputs_shape
    if self.detect_faces:
      # Detect faces
      if self.mode == Mode.BURST:
        faces_rel, _ = self.face_detector(inputs=video[-1:], n_frames=1, fps=fps)
      else:
        faces_rel, _ = self.face_detector(inputs=video, n_frames=inputs_shape[0], fps=fps)
      # If no faces detected: return empty list
      if len(faces_rel) == 0:
        logging.warning("No faces to analyze")
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
      data, unit, conf, note, live = self.rppg(inputs=video,
                                               faces=face,
                                               fps=fps,
                                               override_fps_target=override_fps_target,
                                               override_global_parse=override_global_parse)
      # Parse face results
      face_result = {'face': {
        'coordinates': face,
        'confidence': live,
        'note': "Face detection coordinates for this face, along with live confidence levels."
      }}
      # Parse vital signs results
      vital_signs_results = {}
      for name in self.config['signals']:
        if name in data and name in unit and name in conf and name in note:
          vital_signs_results[name] = {
            f"{'data' if 'waveform' in name else 'value'}": data[name],
            'unit': unit[name],
            'confidence': conf[name],
            'note': note[name]
          }
      if self.estimate_rolling_vitals:
        try:
          if 'ppg_waveform' in self.config['signals']:
            rolling_hr = estimate_hr_from_signal(
              signal=data['ppg_waveform'],
              f_s=fps,
              window_size=CALC_HR_MAX_T,
              scope=EScope.ROLLING,
              method=EMethod.PERIODOGRAM
            )
            rolling_conf = rolling_calc(
              x=conf['ppg_waveform'],
              calc_fn=lambda x: np.nanmean(x),
              min_window_size=CALC_HR_MAX_T,
              max_window_size=CALC_HR_MAX_T,
              overlap=CALC_HR_MAX_T // 2
            )
            vital_signs_results['rolling_heart_rate'] = {
              'data': rolling_hr,
              'unit': 'bpm',
              'confidence': rolling_conf,
              'note': 'Estimate of the rolling heart rate using VitalLens, along with frame-wise confidences between 0 and 1.',
            }
          if 'respiratory_waveform' in self.config['signals']:
            rolling_rr = estimate_rr_from_signal(
              signal=data['respiratory_waveform'],
              f_s=fps,
              window_size=CALC_RR_MAX_T,
              scope=EScope.ROLLING,
              method=EMethod.PERIODOGRAM
            )
            rolling_conf = rolling_calc(
              x=conf['respiratory_waveform'],
              calc_fn=lambda x: np.nanmean(x),
              min_window_size=CALC_RR_MAX_T,
              max_window_size=CALC_RR_MAX_T,
              overlap=CALC_RR_MAX_T // 2
            )
            vital_signs_results['rolling_respiratory_rate'] = {
              'data': rolling_rr,
              'unit': 'bpm',
              'confidence': rolling_conf,
              'note': 'Estimate of the rolling respiratory rate using VitalLens, along with frame-wise confidences between 0 and 1.',
            }
        except ValueError as e:
          logging.debug(f"Issue while computing rolling vitals: {e}")
      face_result['vital_signs'] = vital_signs_results
      face_result['message'] = DISCLAIMER
      results.append(face_result)
    # Export to json
    if self.export_to_json:
      os.makedirs(self.export_dir, exist_ok=True)
      export_filename = f"{export_filename}.json" if export_filename is not None else f"vitallens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
      with open(os.path.join(self.export_dir, export_filename), 'w') as f:
        json.dump(convert_ndarray_to_list(results), f, indent=4)
    return results
  def reset(self):
    """Reset"""
    self.rppg.reset()

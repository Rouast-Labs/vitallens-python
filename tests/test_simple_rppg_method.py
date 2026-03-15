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

import numpy as np
import pytest
import vitallens_core as vc

import sys
sys.path.append('../vitallens-python')

from vitallens.methods.chrom import CHROMRPPGMethod
from vitallens.methods.g import GRPPGMethod
from vitallens.methods.pos import POSRPPGMethod
from vitallens.methods.simple_rppg_method import SimpleRPPGMethod

def test_CHROMRPPGMethod():
  method = CHROMRPPGMethod()
  assert method.method.value == "chrom"
  assert method.session_config.model_name == "chrom"
  assert method.est_window_length == 48
  assert method.est_window_overlap == 47
  assert "heart_rate" in method.session_config.supported_vitals
  rgb_input = np.random.rand(100, 3)
  res = method.algorithm(rgb_input, fps=30.0)
  assert isinstance(res, np.ndarray)
  assert res.shape == (100,)

def test_GRPPGMethod():
  method = GRPPGMethod()
  assert method.method.value == "g"
  assert method.session_config.model_name == "g"
  assert method.est_window_length == 64
  assert method.est_window_overlap == 0
  assert "heart_rate" in method.session_config.supported_vitals
  rgb_input = np.random.rand(100, 3)
  res = method.algorithm(rgb_input, fps=30.0)
  assert isinstance(res, np.ndarray)
  assert res.shape == (100,)

def test_POSRPPGMethod():
  method = POSRPPGMethod()
  assert method.method.value == "pos"
  assert method.session_config.model_name == "pos"
  assert method.est_window_length == 48
  assert method.est_window_overlap == 47
  assert "heart_rate" in method.session_config.supported_vitals
  rgb_input = np.random.rand(100, 3)
  res = method.algorithm(rgb_input, fps=30.0)
  assert isinstance(res, np.ndarray)
  assert res.shape == (100,)

class DummySimpleRPPG(SimpleRPPGMethod):
  def __init__(self):
    super().__init__()
    config = vc.SessionConfig(
      model_name="dummy",
      supported_vitals=["heart_rate"],
      return_waveforms=["ppg_waveform"],
      fps_target=30.0,
      input_size=100,
      n_inputs=1,
      roi_method="face"
    )
    self.parse_config(config, est_window_length=30, est_window_overlap=0)
  def algorithm(self, rgb: np.ndarray, fps: float) -> np.ndarray:
    return np.mean(rgb, axis=-1)

def test_SimpleRPPGMethod_infer_batch(request):
  method = DummySimpleRPPG()
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  test_video_faces = request.getfixturevalue('test_video_faces')
  sig_dict, conf_dict, live = method.infer_batch(
    inputs=test_video_ndarray, faces=test_video_faces, fps=test_video_fps
  )
  assert 'ppg_waveform' in sig_dict
  assert sig_dict['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  assert conf_dict['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  assert live.shape == (test_video_ndarray.shape[0],)

def test_SimpleRPPGMethod_infer_stream():
  method = DummySimpleRPPG()
  frames = np.random.rand(16, 3)
  fps = 30.0
  sig_dict, conf_dict, live, state = method.infer_stream(frames, fps, None)
  assert 'ppg_waveform' in sig_dict
  assert sig_dict['ppg_waveform'].shape == (16,)
  assert conf_dict['ppg_waveform'].shape == (16,)
  assert live.shape == (16,)
  assert state is None
# Copyright (c) 2024 Philipp Rouast
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

import sys
sys.path.append('../vitallens-python')

from vitallens.enums import Mode
from vitallens.methods.chrom import CHROMRPPGMethod
from vitallens.methods.g import GRPPGMethod
from vitallens.methods.pos import POSRPPGMethod
from vitallens.utils import load_config

@pytest.mark.parametrize("override_fps_target", [None, 15])
def test_CHROMRPPGMethod(request, override_fps_target):
  config = load_config("chrom.yaml")
  method = CHROMRPPGMethod(config=config, mode=Mode.BATCH)
  res = method.algorithm(np.random.rand(100, 3), fps=30.)
  assert res.shape == (100,)
  res = method.pulse_filter(res, fps=30.)
  assert res.shape == (100,)
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  test_video_faces = request.getfixturevalue('test_video_faces')
  data, unit, conf, note, live = method(
    inputs=test_video_ndarray, faces=test_video_faces,
    fps=test_video_fps, override_fps_target=override_fps_target)
  assert all(key in data for key in method.signals)
  assert all(key in unit for key in method.signals)
  assert all(key in conf for key in method.signals)
  assert all(key in note for key in method.signals)
  assert data['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  assert conf['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  np.testing.assert_equal(conf['ppg_waveform'], np.ones((test_video_ndarray.shape[0],), np.float32))
  assert conf['heart_rate'] == 1.0
  np.testing.assert_equal(live, np.ones((test_video_ndarray.shape[0],), np.float32))

@pytest.mark.parametrize("override_fps_target", [None, 15])
def test_GRPPGMethod(request, override_fps_target):
  config = load_config("g.yaml")
  method = GRPPGMethod(config=config, mode=Mode.BATCH)
  res = method.algorithm(np.random.rand(100, 3), fps=30.)
  assert res.shape == (100,)
  res = method.pulse_filter(res, fps=30.)
  assert res.shape == (100,)
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  test_video_faces = request.getfixturevalue('test_video_faces')
  data, unit, conf, note, live = method(
    inputs=test_video_ndarray, faces=test_video_faces,
    fps=test_video_fps, override_fps_target=override_fps_target)
  assert all(key in data for key in method.signals)
  assert all(key in unit for key in method.signals)
  assert all(key in conf for key in method.signals)
  assert all(key in note for key in method.signals)
  assert data['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  assert conf['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  np.testing.assert_equal(conf['ppg_waveform'], np.ones((test_video_ndarray.shape[0],), np.float32))
  assert conf['heart_rate'] == 1.0
  np.testing.assert_equal(live, np.ones((test_video_ndarray.shape[0],), np.float32))

@pytest.mark.parametrize("override_fps_target", [None, 15])
def test_POSRPPGMethod(request, override_fps_target):
  config = load_config("pos.yaml")
  method = POSRPPGMethod(config=config, mode=Mode.BATCH)
  res = method.algorithm(np.random.rand(100, 3), fps=30.)
  assert res.shape == (100,)
  res = method.pulse_filter(res, fps=30.)
  assert res.shape == (100,)
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  test_video_faces = request.getfixturevalue('test_video_faces')
  data, unit, conf, note, live = method(
    inputs=test_video_ndarray, faces=test_video_faces,
    fps=test_video_fps, override_fps_target=override_fps_target)
  assert all(key in data for key in method.signals)
  assert all(key in unit for key in method.signals)
  assert all(key in conf for key in method.signals)
  assert all(key in note for key in method.signals)
  assert data['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  assert conf['ppg_waveform'].shape == (test_video_ndarray.shape[0],)
  np.testing.assert_equal(conf['ppg_waveform'], np.ones((test_video_ndarray.shape[0],), np.float32))
  assert conf['heart_rate'] == 1.0
  np.testing.assert_equal(live, np.ones((test_video_ndarray.shape[0],), np.float32))

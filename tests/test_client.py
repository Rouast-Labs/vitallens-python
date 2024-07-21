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

import json
import numpy as np
import os
import pytest

import sys
sys.path.append('../vitallens-python')

from vitallens.client import VitalLens, Method

@pytest.mark.parametrize("method", [Method.G, Method.CHROM, Method.POS])
@pytest.mark.parametrize("detect_faces", [True, False])
@pytest.mark.parametrize("file", [True, False])
@pytest.mark.parametrize("export", [True, False])
def test_VitalLens(request, method, detect_faces, file, export):
  vl = VitalLens(method=method, detect_faces=detect_faces, export_to_json=export)
  if file:
    test_video_path = request.getfixturevalue('test_video_path')
    result = vl(test_video_path, faces = None if detect_faces else [247, 57, 440, 334], export_filename="test")
  else:
    test_video_ndarray = request.getfixturevalue('test_video_ndarray')
    test_video_fps = request.getfixturevalue('test_video_fps')
    result = vl(test_video_ndarray, fps=test_video_fps, faces = None if detect_faces else [247, 57, 440, 334], export_filename="test")
  assert len(result) == 1
  assert result[0]['face']['coordinates'].shape == (360, 4)
  assert result[0]['face']['confidence'].shape == (360,)
  assert result[0]['vital_signs']['ppg_waveform']['data'].shape == (360,)
  assert result[0]['vital_signs']['ppg_waveform']['confidence'].shape == (360,)
  np.testing.assert_allclose(result[0]['vital_signs']['heart_rate']['value'], 60, atol=10)
  assert result[0]['vital_signs']['heart_rate']['confidence'] == 1.0
  if export:
    test_json_path = os.path.join("test.json")
    assert os.path.exists(test_json_path)
    with open(test_json_path, 'r') as f:
      data = json.load(f)
    assert np.asarray(data[0]['face']['coordinates']).shape == (360, 4)
    assert np.asarray(data[0]['face']['confidence']).shape == (360,)
    assert np.asarray(data[0]['vital_signs']['ppg_waveform']['data']).shape == (360,)
    assert np.asarray(data[0]['vital_signs']['ppg_waveform']['confidence']).shape == (360,)
    np.testing.assert_allclose(data[0]['vital_signs']['heart_rate']['value'], 60, atol=10)
    assert data[0]['vital_signs']['heart_rate']['confidence'] == 1.0
    os.remove(test_json_path)

def test_VitalLens_API(request):
  api_key = request.getfixturevalue('test_dev_api_key')
  vl = VitalLens(method=Method.VITALLENS, api_key=api_key, detect_faces=True, export_to_json=False)
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  result = vl(test_video_ndarray, fps=test_video_fps, faces=None, export_filename="test")
  assert len(result) == 1
  assert result[0]['face']['coordinates'].shape == (360, 4)
  assert result[0]['vital_signs']['ppg_waveform']['data'].shape == (360,)
  assert result[0]['vital_signs']['ppg_waveform']['confidence'].shape == (360,)
  assert result[0]['vital_signs']['respiratory_waveform']['data'].shape == (360,)
  assert result[0]['vital_signs']['respiratory_waveform']['confidence'].shape == (360,)
  np.testing.assert_allclose(result[0]['vital_signs']['heart_rate']['value'], 60, atol=0.5)
  np.testing.assert_allclose(result[0]['vital_signs']['heart_rate']['confidence'], 1.0, atol=0.1)
  np.testing.assert_allclose(result[0]['vital_signs']['respiratory_rate']['value'], 13.5, atol=0.5)
  np.testing.assert_allclose(result[0]['vital_signs']['respiratory_rate']['confidence'], 1.0, atol=0.1)
  assert not os.path.exists("test.json")

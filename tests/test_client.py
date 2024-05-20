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

from vitallens.client import VitalLens, Method

@pytest.mark.parametrize("method", [Method.G, Method.CHROM, Method.POS])
@pytest.mark.parametrize("detect_faces", [True, False])
@pytest.mark.parametrize("file", [True, False])
def test_VitalLens(request, method, detect_faces, file):
  vl = VitalLens(method=method, detect_faces=detect_faces)
  if file:
    test_video_path = request.getfixturevalue('test_video_path')
    result = vl(test_video_path, faces = None if detect_faces else [425, 116, 671, 433])
  else:
    test_video_ndarray = request.getfixturevalue('test_video_ndarray')
    test_video_fps = request.getfixturevalue('test_video_fps')
    result = vl(test_video_ndarray, fps=test_video_fps, faces = None if detect_faces else [425, 116, 671, 433])
  assert len(result) == 1
  assert result[0]['face'].shape == (139, 4)
  assert result[0]['pulse']['val'].shape == (139,)
  np.testing.assert_allclose(result[0]['hr']['val'], 71.5, atol=2)

def test_VitalLens_API(request):
  api_key = request.getfixturevalue('test_dev_api_key')
  vl = VitalLens(method=Method.VITALLENS, api_key=api_key, detect_faces=True)
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  result = vl(test_video_ndarray, fps=test_video_fps, faces=None)
  assert len(result) == 1
  assert result[0]['face'].shape == (139, 4)
  assert result[0]['pulse']['val'].shape == (139,)
  assert result[0]['pulse']['conf'].shape == (139,)
  assert result[0]['resp']['val'].shape == (139,)
  assert result[0]['resp']['conf'].shape == (139,)
  np.testing.assert_allclose(result[0]['hr']['val'], 73, atol=0.5)
  np.testing.assert_allclose(result[0]['rr']['val'], 15, atol=0.5)

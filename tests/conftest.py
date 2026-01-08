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
import os
from prpy.ffmpeg.probe import probe_video
from prpy.ffmpeg.readwrite import read_video_from_path
import pytest

import sys
sys.path.append('../vitallens-python')

from vitallens.ssd import FaceDetector
from vitallens.utils import download_file

TEST_VIDEO_URL = "https://github.com/Rouast-Labs/vitallens-python/raw/main/examples/sample_video_2.mp4"
TEST_VIDEO_PATH = "examples/sample_video_2.mp4"

# Download the test video before running any tests
download_file(TEST_VIDEO_URL, TEST_VIDEO_PATH)

@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
  os.environ.setdefault("VITALLENS_API_ORIGIN", "test")

@pytest.fixture(scope='session')
def test_video_path():
  return TEST_VIDEO_PATH

@pytest.fixture(scope='session')
def test_video_ndarray():
  video, _ = read_video_from_path(path=TEST_VIDEO_PATH, pix_fmt='rgb24')
  return video

@pytest.fixture(scope='session')
def test_video_fps():
  fps, *_ = probe_video(TEST_VIDEO_PATH)
  return fps

@pytest.fixture(scope='session')
def test_video_shape():
  _, n, w, h, *_ = probe_video(TEST_VIDEO_PATH)
  return (n, h, w, 3)

@pytest.fixture(scope='session')
def test_video_faces(request):
  det = FaceDetector(
    max_faces=1, fs=1.0, iou_threshold=0.45, score_threshold=0.9)
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  test_video_fps = request.getfixturevalue('test_video_fps')
  boxes, _ = det(test_video_ndarray,
                 n_frames=test_video_ndarray.shape[0],
                 fps=test_video_fps)
  boxes = (boxes * [test_video_ndarray.shape[2], test_video_ndarray.shape[1], test_video_ndarray.shape[2], test_video_ndarray.shape[1]]).astype(int)
  return boxes[:,0].astype(np.int64)

@pytest.fixture(scope='session')
def test_dev_api_key():
  api_key = os.getenv('VITALLENS_DEV_API_KEY')
  if not api_key:
    raise pytest.UsageError(
        "VITALLENS_DEV_API_KEY environment variable is not set. Please set this variable "
        "to a valid VitalLens API Key to run the tests. You can do this by exporting the "
        "variable in your shell or adding it to your conda environment configuration."
    )
  return api_key

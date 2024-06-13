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

from vitallens.client import Method
from vitallens.utils import load_config, probe_video_inputs, parse_video_inputs
from vitallens.utils import merge_faces, check_faces

@pytest.mark.parametrize("method", [m for m in Method])
def test_load_config(method):
  config = load_config(method.name.lower() + ".yaml")
  assert config['model'] == method.name.lower()

@pytest.mark.parametrize("file", [True, False])
def test_probe_video_inputs(request, file):
  if file:
    test_video_path = request.getfixturevalue('test_video_path')
    video_shape, fps = probe_video_inputs(test_video_path)
  else:
    test_video_ndarray = request.getfixturevalue('test_video_ndarray')
    test_video_fps = request.getfixturevalue('test_video_fps')
    video_shape, fps = probe_video_inputs(test_video_ndarray, fps=test_video_fps)
  assert video_shape == (360, 480, 768, 3)
  assert fps == 30

def test_probe_video_inputs_no_file():
  with pytest.raises(Exception):
    _ = probe_video_inputs("does_not_exist", fps="fps")

def test_probe_video_inputs_wrong_fps(request):
  with pytest.raises(Exception):
    test_video_path = request.getfixturevalue('test_video_path')
    _ = probe_video_inputs(test_video_path, fps="fps")

def test_probe_video_inputs_no_fps(request):
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  with pytest.raises(Exception):
    _ = probe_video_inputs(test_video_ndarray)

def test_probe_video_inputs_wrong_dtype(request):
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  with pytest.raises(Exception):
    _ = probe_video_inputs(test_video_ndarray.astype(np.float32), fps=30.)

def test_probe_video_inputs_wrong_shape_1(request):
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  with pytest.raises(Exception):
    _ = probe_video_inputs(test_video_ndarray[np.newaxis], fps=30.)

def test_probe_video_inputs_wrong_shape_2(request):
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  with pytest.raises(Exception):
    _ = probe_video_inputs(test_video_ndarray[...,0:1], fps=30.)

def test_probe_video_inputs_wrong_shape_3(request):
  test_video_ndarray = request.getfixturevalue('test_video_ndarray')
  with pytest.raises(Exception):
    _ = probe_video_inputs(test_video_ndarray[:10], fps=30.)

def test_probe_video_inputs_wrong_type():
  with pytest.raises(Exception):
    _ = probe_video_inputs(12345, fps=30.)

@pytest.mark.parametrize("file", [True, False])
@pytest.mark.parametrize("roi", [None, (200, 0, 500, 350)])
@pytest.mark.parametrize("target_size", [None, 200])
@pytest.mark.parametrize("target_fps", [None, 15])
def test_parse_video_inputs(request, file, roi, target_size, target_fps):
  if file:
    test_video_path = request.getfixturevalue('test_video_path')
    parsed, fps_in, video_shape_in, ds_factor = parse_video_inputs(
      test_video_path, roi=roi, target_size=target_size, target_fps=target_fps)
  else:
    test_video_ndarray = request.getfixturevalue('test_video_ndarray')
    test_video_fps = request.getfixturevalue('test_video_fps')
    parsed, fps_in, video_shape_in, ds_factor = parse_video_inputs(
      test_video_ndarray, fps=test_video_fps, roi=roi, target_size=target_size,
      target_fps=target_fps)
  assert parsed.shape == (360 if target_fps is None else 360 // 2,
                          200 if target_size is not None else (350 if roi is not None else 480),
                          200 if target_size is not None else (300 if roi is not None else 768),
                          3)
  assert fps_in == 30
  assert video_shape_in == (360, 480, 768, 3)
  assert ds_factor == 1 if target_fps is None else 2

def test_parse_video_inputs_no_file():
  with pytest.raises(Exception):
    _ = parse_video_inputs("does_not_exist")

def test_parse_video_inputs_wrong_type():
  with pytest.raises(Exception):
    _ = parse_video_inputs(12345, fps=30.)

def test_merge_faces():
  np.testing.assert_equal(
    merge_faces(np.asarray([[2, 4, 3, 7],
                            [1, 3, 4, 6],
                            [2, 4, 5, 7],
                            [3, 5, 4, 8]])),
    np.asarray([1, 3, 5, 8]))
  
@pytest.mark.parametrize("faces", [None, [[[3, 5, 4, 8], [3, 5, 4, 8]], [[3, 5, 4, 8], [2, 6, 4, 8]]], [[3, 5, 4, 8], [2, 6, 4, 8]], [3, 5, 4, 8]])
@pytest.mark.parametrize("faces_list", [True, False])
def test_check_faces(faces, faces_list):
  if faces is None and not faces_list:
    pytest.skip("Skip because parameter combination does not work")
  if faces_list:
    faces = check_faces(faces, inputs_shape=(2, 10, 10, 3))
  else:
    faces = check_faces(np.asarray(faces), inputs_shape=(2, 10, 10, 3))
  assert isinstance(faces, np.ndarray)
  assert faces.ndim == 3
  assert faces.shape[1] == 2
  assert faces.shape[2] == 4

def test_check_faces_not_flat_point_form():
  with pytest.raises(Exception):
    _ = check_faces([0, 0, 0, 0, 0], inputs_shape=(2, 10, 10, 3))

def test_check_faces_n_frames_n_dets_not_matching_1():
  with pytest.raises(Exception):
    _ = check_faces([[3, 5, 4, 8]], inputs_shape=(2, 10, 10, 3))

def test_check_faces_n_frames_n_dets_not_matching_2():
  with pytest.raises(Exception):
    _ = check_faces([[[3, 5, 4, 8], [3, 5, 4, 8], [3, 5, 4, 8]]], inputs_shape=(2, 10, 10, 3))

def test_check_faces_invalid_dets():
  with pytest.raises(Exception):
    _ = check_faces([1, 3, 2, 2], inputs_shape=(2, 10, 10, 3))
    
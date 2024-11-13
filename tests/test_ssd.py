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

from vitallens.ssd import nms, enforce_temporal_consistency
from vitallens.ssd import interpolate_unscanned_frames, FaceDetector

@pytest.mark.parametrize("max_output_size", [1, 2, 3])
def test_nms(max_output_size):
  boxes = np.asarray([[[0.0, 0.0, 0.25, 0.25],
                       [0.5, 0.0, 1.0, 0.5],
                       [0.5, 0.125, 0.875, 0.5],
                       [0.25, 0.5, 0.5, 0.75],
                       [0.375, 0.625, 0.625, 0.875],
                       [0.5, 0.75, 0.75, 1.0]],
                      [[0.0, 0.0, 0.25, 0.25],
                       [0.5, 0.0, 1.0, 0.5],
                       [0.5, 0.125, 0.875, 0.5],
                       [0.25, 0.5, 0.5, 0.75],
                       [0.375, 0.625, 0.625, 0.875],
                       [0.5, 0.75, 0.75, 1.0]]])
  scores = np.asarray([[0.01,
                        0.99,
                        0.95,
                        0.97,
                        0.80,
                        0.09],
                       [0.01,
                        0.99,
                        0.95,
                        0.97,
                        0.80,
                        0.98]])
  if max_output_size == 1:
    idxs = np.asarray([[1], [1]])
    num_valid = np.asarray([1, 1])
  elif max_output_size == 2:
    idxs = np.asarray([[1, 3], [1, 5]])
    num_valid = np.asarray([2, 2])
  else:
    idxs = np.asarray([[1, 3, 0], [1, 5, 3]])
    num_valid = np.asarray([2, 3])
  out_idxs, out_num_valid = nms(boxes=boxes, scores=scores, max_output_size=max_output_size, iou_threshold=0.45, score_threshold=0.9)
  np.testing.assert_equal(
    out_idxs,
    idxs)
  np.testing.assert_equal(
    out_num_valid,
    num_valid)

def test_enforce_temporal_consistency():
  # Example with 2 moving faces, 5 time steps, no detection for face 1 in time step 2, faces swapped in time step 4
  boxes = np.array(
    [[[.125, .125, .375, .375], [.125, .5, .375, .75]],
     [[.125, .25,  .375, .5  ], [.25,  .5, .5,   .75]],
     [[.125, .375, .375, .625], [.375, .5, .625, .75]],
     [[.5,   .5,   .75,  .75 ], [.125, .5, .375, .75]],
     [[.125, .625, .375, .875], [.625, .5, .875, .75]]])
  info = np.array(
    [[[0, 1, 1, .99], [0, 1, 1, .99]],
     [[1, 1, 0, .2 ], [1, 1, 1, .99]],
     [[2, 1, 1, .99], [2, 1, 1, .99]],
     [[3, 1, 1, .99], [3, 1, 1, .99]],
     [[4, 1, 1, .99], [4, 1, 1, .99]]])
  boxes_out, info_out = enforce_temporal_consistency(
    boxes=boxes, info=info, n_frames=5)
  np.testing.assert_equal(
    boxes_out,
      np.array(
        [[[.125, .5, .375, .75], [.125, .125, .375, .375]],
         [[.25,  .5, .5,   .75], [.125, .25,  .375, .5  ]],
         [[.375, .5, .625, .75], [.125, .375, .375, .625]],
         [[.5,   .5, .75,  .75], [.125, .5,   .375, .75 ]],
         [[.625, .5, .875, .75], [.125, .625, .375, .875]]]))
  np.testing.assert_equal(
    info_out,
      np.array(
        [[[0, 1, 1, .99], [0, 1, 1, .99]],
         [[1, 1, 1, .99], [1, 1, 0, .2 ]],
         [[2, 1, 1, .99], [2, 1, 1, .99]],
         [[3, 1, 1, .99], [3, 1, 1, .99]],
         [[4, 1, 1, .99], [4, 1, 1, .99]]]))
  
def test_interpolate_unscanned_frames():
  # Example with 2 moving faces, 3 time steps, no detection for face 1 in time step 2, faces swapped in time step 4
  boxes = np.array(
    [[[.125, .5, .375, .75], [.125, .125, .375, .375]],
     [[.25,  .5, .5,   .75], [.125, .25,  .375, .5  ]],
     [[.375, .5, .625, .75], [.125, .375, .375, .625]]])
  info = np.array(
    [[[0, 1, 1, .99], [0, 1, 1, .99]],
     [[2, 1, 1, .99], [2, 1, 0, .2 ]],
     [[4, 1, 1, .99], [4, 1, 1, .99]]])
  boxes_out, info_out = interpolate_unscanned_frames(
    boxes=boxes, info=info, n_frames=5)
  np.testing.assert_equal(
    boxes_out,
      np.array(
        [[[.125,  .5, .375,  .75], [.125, .125,  .375, .375 ]],
         [[.1875, .5, .4375, .75], [.125, .1875, .375, .4375]],
         [[.25,   .5, .5,    .75], [.125, .25,   .375, .5   ]],
         [[.3125, .5, .5625, .75], [.125, .3125, .375, .5625]],
         [[.375,  .5, .625,  .75], [.125, .375,  .375, .625 ]]]))
  np.testing.assert_equal(
    info_out,
      np.array(
        [[[0, 1, 1, .99], [0, 1, 1, .99]],
         [[1, 0, 0, 0  ], [1, 0, 0, 0  ]], # Imperfection of the implementation
         [[2, 1, 1, .99], [2, 1, 0, .2 ]],
         [[3, 0, 0, 0  ], [3, 0, 0, 0  ]],
         [[4, 1, 1, .99], [4, 1, 1, .99]]]))

@pytest.mark.parametrize("file", [True, False])
def test_FaceDetector(request, file):
  det = FaceDetector(
    max_faces=2, fs=1.0, iou_threshold=0.45, score_threshold=0.9)
  if file:
    test_video_path = request.getfixturevalue('test_video_path')
    test_video_shape = request.getfixturevalue('test_video_shape')
    test_video_fps = request.getfixturevalue('test_video_fps')
    boxes, info = det(inputs=test_video_path,
                      n_frames=test_video_shape[0],
                      fps=test_video_fps)
  else:
    test_video_ndarray = request.getfixturevalue('test_video_ndarray')
    test_video_fps = request.getfixturevalue('test_video_fps')
    boxes, info = det(inputs=test_video_ndarray,
                      n_frames=test_video_ndarray.shape[0],
                      fps=test_video_fps)
  assert boxes.shape == (360, 1, 4)
  assert info.shape == (360, 1, 4)
  np.testing.assert_allclose(boxes[0,0],
                             [0.32223, 0.118318, 0.572684, 0.696835],
                             atol=0.01)
  
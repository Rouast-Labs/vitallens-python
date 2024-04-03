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
from propy.ffmpeg.probe import probe_video
from propy.ffmpeg.readwrite import read_video_from_path
from propy.numpy.image import crop_slice_resize
from typing import Union, Tuple
import os

MIN_N_FRAMES = 16

def probe_inputs(
    inputs: Union[np.ndarray, str],
    fps: float = None
  ) -> Tuple[tuple, float]:
  """TODO"""
  # Check if input is array or file name
  if isinstance(inputs, str):
    if os.path.isfile(inputs):
      try:
        fps, n_frames, width, height, *_ = probe_video(inputs)
        return (n_frames, height, width, 3), fps
      except Exception as e:
        raise ValueError("Problem probing video at {}: {}".format(inputs, e))
    else:
      raise ValueError("No file found at {}".format(inputs))
  elif isinstance(inputs, np.ndarray):
    if fps is None:
      raise ValueError("fps must be specified for ndarray input")
    if inputs.dtype != np.uint8:
      raise ValueError("inputs.dtype should be uint8, but found {}".format(inputs.dtype))
    if len(inputs.shape) != 4 or inputs.shape[0] < MIN_N_FRAMES or inputs.shape[3] != 3:
      raise ValueError("inputs should have shape (n_frames [>= {}], h, w, 3), but found {}".format(MIN_N_FRAMES, inputs.shape))
    return inputs.shape, fps
  else:
    raise ValueError("Invalid inputs {}, type {}".format(inputs, type(input)))

def parse_inputs(
    inputs: Union[np.ndarray, str],
    fps: float = None,
    roi: tuple = None,
    target_size: tuple = None,
  ) -> Tuple[np.ndarray, float]:
  """Parse user inputs.

  Args:
    inputs: The inputs. Either a filepath to video file or ndarray
    fps: Frames per second of inputs. Can be `None` if video file provided.
    roi: The region of interest as [x0, y0, x1, y1]. Use None to keep all.
    target_size: Optional target size as tuple (h, w)
  Returns:
    inputs: Inputs as `np.ndarray` with type uint8. Shape (n, h, w, c)
      if target_size provided, w = target_size[0] and h = target_size[1]
    fps: Frames per second of inputs
  """
  # Check if input is array or file name
  if isinstance(inputs, str):
    if os.path.isfile(inputs):
      try:
        if fps is None: fps, *_ = probe_video(inputs)
        if roi is not None: roi = (roi[0], roi[1], roi[2]-roi[0], roi[3]-roi[1])
        if target_size is not None: target_size = (target_size[1], target_size[0])
        inputs, _ = read_video_from_path(
          path=inputs, crop=roi, scale=target_size, pix_fmt='rgb24')
        return inputs, fps
      except Exception as e:
        raise ValueError("Problem reading video from {}: {}".format(inputs, e))
    else:
      raise ValueError("No file found at {}".format(inputs))
  elif isinstance(inputs, np.ndarray):
    if fps is None:
      raise ValueError("fps must be specified for ndarray input")
    if inputs.dtype != np.uint8:
      raise ValueError("inputs.dtype should be uint8, but found {}".format(inputs.dtype))
    if len(inputs.shape) != 4 or inputs.shape[0] < MIN_N_FRAMES or inputs.shape[3] != 3:
      raise ValueError("inputs should have shape (n_frames [>= {}], h, w, 3), but found {}".format(MIN_N_FRAMES, inputs.shape))
    # Crop / scale if necessary
    if roi is not None or target_size is not None:
      if target_size is None: target_size = (inputs.shape[1], inputs.shape[2])
      inputs = crop_slice_resize(
        inputs=inputs, target_size=target_size, roi=roi, preserve_aspect_ratio=False,
        library='PIL')
    return inputs, fps
  else:
    raise ValueError("Invalid inputs {}, type {}".format(inputs, type(input)))

def check_faces(
    faces: Union[list, np.ndarray],
    inputs_shape: tuple
  ) -> np.ndarray:
  """Make sure the face detections are in a correct format.

  Args:
    faces: The specified faces in form [x0, y0, x1, y1]. Either
      - list/ndarray of shape (n_faces, n_frames, 4) for multiple faces detected on multiple frames
      - list/ndarray of shape (n_frames, 4) for single face detected on mulitple frames
      - list/ndarray of shape (4,) for single face detected globally
      - None to assume frames already cropped to single face
    inputs_shape: The shape of the inputs.
  Returns:
    faces: The faces. np.ndarray of shape (n_faces, n_frames, 4) in form [x_0, y_0, x_1, y_1]
  """
  n_frames, h, w, _ = inputs_shape
  if faces is None:
    # Assume that each entire frame is a single face
    logging.info("No faces given - assuming that frames have been cropped to a single face")
    faces = np.tile(np.asarray([0, 0, w, h]), (n_frames, 1))[np.newaxis] # (1, n_frames, 4)
  else:
    assert faces.shape[-1] == 4, "Face detections must be in flat point form"
    if len(faces.shape) == 1:
      # Single face detection given - repeat for n_frames
      faces = np.tile(faces, (n_frames, 1))[np.newaxis] # (1, n_frames, 4)
    elif len(faces.shape) == 2:
      # Single face detections for multiple frames given
      assert faces.shape[0] == n_frames, "Number of frames must match for inputs and face detections"
      faces = faces[np.newaxis]
    elif len(faces.shape) == 3:
      if faces.shape[1] == 1:
        # Multiple face detections for single frame given
        faces = np.tile(faces, (1, n_frames, 1)) # (n_faces, n_frames, 4)
      else:
        # Multiple face detections for multiple frames given
        assert faces.shape[1] == n_frames, "Number of frames must match for inputs and face detections"
  # Check that x0 < x1 and y0 < y1 for all faces
  assert np.all((faces[...,2] - faces[...,0]) > 0), "Face detections are invalid, should be in form [x0, y0, x1, y1]"
  return faces

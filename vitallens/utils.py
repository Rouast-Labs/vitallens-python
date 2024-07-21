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

import importlib.resources
import logging
import numpy as np
import os
from prpy.ffmpeg.probe import probe_video
from prpy.ffmpeg.readwrite import read_video_from_path
from prpy.numpy.image import crop_slice_resize
from typing import Union, Tuple
import urllib.request
import yaml

from vitallens.constants import API_MIN_FRAMES

def load_config(filename: str) -> dict:
  """Load a yaml config file.

  Args:
    filename: The filename of the yaml config file
  Returns:
    loaded: The contents of the yaml config file
  """
  with importlib.resources.open_binary('vitallens.configs', filename) as f:
    loaded = yaml.load(f, Loader=yaml.Loader)
  return loaded

def download_file(url: str, dest: str):
  """Download a file if necessary.

  Args:
    url: The url to download the file from
    dest: The path to write the downloaded file to
  """
  if not os.path.exists(dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logging.info("Downloading {} to {}".format(url, dest))
    urllib.request.urlretrieve(url, dest)
  else:
    logging.info("{} already exists, skipping download.".format(dest))

def probe_video_inputs(
    video: Union[np.ndarray, str],
    fps: float = None
  ) -> Tuple[tuple, float]:
  """Check the video inputs and probe to extract metadata.

  Args:
    video: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
      with a sequence of frames in unscaled uint8 RGB format, or a path to a
      video file.
    fps: Sampling frequency of the input video. Required if type(video)==np.ndarray.
  Returns:
    video_shape: The shape of the input video as (n_frames, h, w, 3)
    fps: Sampling frequency of the input video.
  """
  # Check that fps is correct type
  if not (fps is None or isinstance(fps, (int, float))):
    raise ValueError("fps should be a number, but got {}".format(type(fps)))
  # Check if video is array or file name
  if isinstance(video, str):
    if os.path.isfile(video):
      try:
        fps_, n, w_, h_, _, _, r = probe_video(video)
        if fps is None: fps = fps_
        if abs(r) == 90: h = w_; w = h_
        else: h = h_; w = w_
        return (n, h, w, 3), fps
      except Exception as e:
        raise ValueError("Problem probing video at {}: {}".format(video, e))
    else:
      raise ValueError("No file found at {}".format(video))
  elif isinstance(video, np.ndarray):
    if fps is None:
      raise ValueError("fps must be specified for ndarray input")
    if video.dtype != np.uint8:
      raise ValueError("video.dtype should be uint8, but got {}".format(video.dtype))
    if len(video.shape) != 4 or video.shape[0] < API_MIN_FRAMES or video.shape[3] != 3:
      raise ValueError("video should have shape (n_frames [>= {}], h, w, 3), but found {}".format(API_MIN_FRAMES, video.shape))
    return video.shape, fps
  else:
    raise ValueError("Invalid video {}, type {}".format(video, type(input)))

def parse_video_inputs(
    video: Union[np.ndarray, str],
    fps: float = None,
    roi: tuple = None,
    target_size: Union[int, tuple] = None,
    target_fps: float = None,
    library: str = 'prpy',
    scale_algorithm: str = 'bilinear',
    trim: tuple = None
  ) -> Tuple[np.ndarray, float, tuple, int]:
  """Parse video inputs into required shape.

  Args:
    video: The video input. Either a filepath to video file or ndarray
    fps: Framerate of video input. Can be `None` if video file provided.
    roi: The region of interest as (x0, y0, x1, y1). Use None to keep all.
    target_size: Optional target size as int or tuple (h, w)
    target_fps: Optional target framerate
    library: Library to use for resample if video is np.ndarray
    scale_algorithm: Algorithm to use for resample
    trim: Frame numbers for temporal trimming (start, end) (optional).
  Returns:
    parsed: Parsed inputs as `np.ndarray` with type uint8. Shape (n, h, w, c)
      if target_size provided, h = target_size[0] and w = target_size[1].
    fps_in: Frame rate of original inputs
    shape_in: Shape of original inputs in form (n, h, w, c)
    ds_factor: Temporal downsampling factor applied
    idxs: The frame indices returned from original video
  """
  # Check if input is array or file name
  if isinstance(video, str):
    if os.path.isfile(video):
      try:
        fps_, n, w_, h_, _, _, r = probe_video(video)
        if fps is None: fps = fps_
        if roi is not None: roi = (int(roi[0]), int(roi[1]), int(roi[2]-roi[0]), int(roi[3]-roi[1]))
        if isinstance(target_size, tuple): target_size = (target_size[1], target_size[0])
        if abs(r) == 90: h = w_; w = h_
        else: h = h_; w = w_
        video, ds_factor = read_video_from_path(
          path=video, target_fps=target_fps, crop=roi, scale=target_size, trim=trim,
          pix_fmt='rgb24', dim_deltas=(1,1,1), scale_algorithm=scale_algorithm)
        start_idx = max(0, trim[0]) if trim is not None else 0
        end_idx = min(n, trim[1]) if trim is not None else n
        idxs = list(range(start_idx, end_idx, ds_factor))
        return video, fps, (n, h, w, 3), ds_factor, idxs
      except Exception as e:
        raise ValueError("Problem reading video from {}: {}".format(video, e))
    else:
      raise ValueError("No file found at {}".format(video))
  elif isinstance(video, np.ndarray):
    video_shape_in = video.shape
    # Downsample / crop / scale if necessary
    ds_factor = 1
    if target_fps is not None:
      if target_fps > fps: logging.warning("target_fps should not be greater than fps. Ignoring.")
      else: ds_factor = max(round(fps / target_fps), 1)
    target_idxs = None if ds_factor == 1 else list(range(video.shape[0])[0::ds_factor])
    if trim is not None:
      if target_idxs is None: target_idxs = range(video_shape_in[0])
      target_idxs = [idx for idx in target_idxs if trim[0] <= idx < trim[1]]
    if roi is not None or target_size is not None or target_idxs is not None:
      if target_size is None and roi is not None: target_size = (int(roi[3]-roi[1]), int(roi[2]-roi[0]))
      elif target_size is None: target_size = (video.shape[1], video.shape[2])
      video = crop_slice_resize(
        inputs=video, target_size=target_size, roi=roi, target_idxs=target_idxs,
        preserve_aspect_ratio=False, library=library, scale_algorithm=scale_algorithm)
    if target_idxs is None: target_idxs = list(range(video_shape_in[0]))
    return video, fps, video_shape_in, ds_factor, target_idxs
  else:
    raise ValueError("Invalid video {}, type {}".format(video, type(video)))

def merge_faces(faces: np.ndarray) -> tuple:
  """Compute the union of all faces.
  
  Args:
    faces: The face detections. Shape (n, 4)
  Returns:
    union: Tuple (x0, y0, x1, y1)
  """
  # Get the minimum x0, y0 and maximum x1, y1 values across all frames
  x0 = np.min(faces[:, 0])
  y0 = np.min(faces[:, 1])
  x1 = np.max(faces[:, 2])
  y1 = np.max(faces[:, 3])
  # Return as tuple
  return (x0, y0, x1, y1)

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
    faces = np.tile(np.asarray([0, 0, w, h], dtype=np.int64), (n_frames, 1))[np.newaxis] # (1, n_frames, 4)
  else:
    faces = np.asarray(faces, dtype=np.int64)
    if faces.shape[-1] != 4: raise ValueError("Face detections must be in flat point form")
    if len(faces.shape) == 1:
      # Single face detection given - repeat for n_frames
      faces = np.tile(faces, (n_frames, 1))[np.newaxis] # (1, n_frames, 4)
    elif len(faces.shape) == 2:
      # Single face detections for multiple frames given
      if faces.shape[0] != n_frames:
        raise ValueError("Assuming detections of a single face for multiple frames given, but number of frames ({}) did not match number of face detections ({})".format(
          n_frames, faces.shape[0]))
      faces = faces[np.newaxis]
    elif len(faces.shape) == 3:
      if faces.shape[1] == 1:
        # Multiple face detections for single frame given
        faces = np.tile(faces, (1, n_frames, 1)) # (n_faces, n_frames, 4)
      else:
        # Multiple face detections for multiple frames given
        if faces.shape[1] != n_frames:
          raise ValueError("Assuming detections of multiple faces for multiple frames given, but number of frames ({}) did not match number of detections for each face ({})".format(
            n_frames, faces.shape[1]))
  # Check that x0 < x1 and y0 < y1 for all faces
  if not (np.all((faces[...,2] - faces[...,0]) > 0) and np.all((faces[...,3] - faces[...,1]) > 0)):
    raise ValueError("Face detections are invalid, should be in form [x0, y0, x1, y1]")
  return faces

def convert_ndarray_to_list(d: Union[dict, list, np.ndarray]):
  """Recursively convert any np.ndarray to list in nested object.
  
  Args:
    d: Nested object consisting of list, dict, and np.ndarray
  Returns:
    out: The same object with any np.ndarray converted to list
  """
  if isinstance(d, np.ndarray):
    return d.tolist()
  elif isinstance(d, dict):
    return {k: convert_ndarray_to_list(v) for k, v in d.items()}
  elif isinstance(d, list):
    return [convert_ndarray_to_list(i) for i in d]
  else:
    return d
  
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
import os
from typing import Union
import urllib.request

def download_file(url: str, dest: str):
  """Download a file if necessary.

  Args:
    url: The url to download the file from
    dest: The path to write the downloaded file to
  """
  if not os.path.exists(dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    logging.info(f"Downloading {url} to {dest}")
    urllib.request.urlretrieve(url, dest)
  else:
    logging.info(f"{dest} already exists, skipping download.")

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
        raise ValueError(f"Assuming detections of a single face for multiple frames given, but number of frames ({n_frames}) did not match number of face detections ({faces.shape[0]})")
      faces = faces[np.newaxis]
    elif len(faces.shape) == 3:
      if faces.shape[1] == 1:
        # Multiple face detections for single frame given
        faces = np.tile(faces, (1, n_frames, 1)) # (n_faces, n_frames, 4)
      else:
        # Multiple face detections for multiple frames given
        if faces.shape[1] != n_frames:
          raise ValueError(f"Assuming detections of multiple faces for multiple frames given, but number of frames ({n_frames}) did not match number of detections for each face ({faces.shape[1]})")
  # Check that x0 < x1 and y0 < y1 for all faces
  if not (np.all((faces[...,2] - faces[...,0]) > 0) and np.all((faces[...,3] - faces[...,1]) > 0)):
    raise ValueError("Face detections are invalid, should be in form [x0, y0, x1, y1]")
  return faces

def check_faces_in_roi(
    faces: np.ndarray,
    roi: Union[np.ndarray, tuple, list],
    percentage_required_inside_roi: tuple = (0.5, 0.5)
  ) -> bool:
  """Check whether all faces are sufficiently inside the ROI.

  Args:
    faces: The faces. Shape (n_faces, 4) in form (x0, y0, x1, y1)
    roi: The region of interest. Shape (4,) in form (x0, y0, x1, y1)
    percentage_required_inside_roi: Tuple (w, h) indicating what percentage
      of width/height of face is required to remain inside the ROI.
  Returns:
    out: True if all faces are sufficiently inside the ROI.
  """
  faces_w = faces[:,2] - faces[:,0]
  faces_h = faces[:,3] - faces[:,1]
  faces_inside_roi = np.logical_and(
    np.logical_and(faces[:,2] - roi[0] > percentage_required_inside_roi[0] * faces_w,
                   roi[2] - faces[:,0] > percentage_required_inside_roi[0] * faces_w),
    np.logical_and(faces[:,3] - roi[1] > percentage_required_inside_roi[1] * faces_h,
                   roi[3] - faces[:,1] > percentage_required_inside_roi[1] * faces_h))
  facess_inside_roi = np.all(faces_inside_roi)
  return facess_inside_roi

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
  
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

import itertools
import logging
import math
import numpy as np
import os
from prpy.numpy.signal import interpolate_vals
import sys
from typing import Tuple

if sys.version_info >= (3, 9):
  from importlib.resources import files
else:
  from importlib_resources import files

from vitallens.utils import parse_video_inputs

INPUT_SIZE = (240, 320)
MAX_SCAN_FRAMES = 60

def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    max_output_size: int,
    iou_threshold: float,
    score_threshold: float
  ) -> Tuple[np.ndarray, np.ndarray]:
  """Non-maximum suppression for a batch of box predictions.

  Args:
    boxes: Boxes in point form [0, 1]. Shape (n_batch, n_anchors, 4)
    scores: Probability scores. Shape (n_batch, n_anchors)
    max_output_size: Maximum number of boxes to be selected. Scalar.
    iou_threshold: Threshold wrt iou for amount of box overlap. Scalar.
    score_threshold: Threshold wrt score for removing boxes. Scalar.
  Returns:
    idxs: The selected indices padded with zero. Shape (n_batch, max_output_size)
  """
  n_batch = boxes.shape[0]
  # Split up box coordinates
  x1 = boxes[..., 0]
  y1 = boxes[..., 1]
  x2 = boxes[..., 2]
  y2 = boxes[..., 3]
  # Compute the areas for all boxes
  areas = (x2 - x1) * (y2 - y1)
  # Initialise result
  out_idxs = np.zeros((n_batch, max_output_size), dtype=np.int64)
  num_valid = []
  # Do nms separately for each batch element
  for b in range(n_batch):
    # Get sorted idxs for scores high->low
    idxs = scores[b].argsort()[::-1]
    # Iterate through idxs
    result_b = []
    while idxs.size > 0:
      i = idxs[0]
      # Stop when best score falls under threshold
      if scores[b,i] < score_threshold or len(result_b) == max_output_size:
        break
      # Keep best scoring idx
      result_b.append(i)
      # Calculate IOU between this idx and all others
      xx1 = np.maximum(x1[b,i], x1[b,idxs[1:]])
      yy1 = np.maximum(y1[b,i], y1[b,idxs[1:]])
      xx2 = np.minimum(x2[b,i], x2[b,idxs[1:]])
      yy2 = np.minimum(y2[b,i], y2[b,idxs[1:]])
      w = np.maximum(0.0, xx2 - xx1)
      h = np.maximum(0.0, yy2 - yy1)
      inters = w * h
      ious = inters / (areas[b,i] + areas[b,idxs[1:]] - inters)
      # Keep only idxs for which IOU is under threshold
      keep = np.where(ious <= iou_threshold)[0]
      idxs = idxs[keep + 1]
    # Write result_b to result
    out_idxs[b,0:len(result_b)] = result_b
    num_valid.append(len(result_b))
  return out_idxs, np.asarray(num_valid)

def enforce_temporal_consistency(
    boxes: np.ndarray,
    info: np.ndarray,
    n_frames: int
  ) -> Tuple[np.ndarray, np.ndarray]:
  """Enforce temporal consistency by sorting faces along second axis to minimize spatial distances.
  
  Args:
    boxes: Detected boxes in point form [0, 1], shape (n_frames, n_faces, 4)
    info: Detection info: idx, scanned, scan_found_face, confidence. Shape (n_frames, n_faces, 4)
    n_frames: Number of frames in the original input.
  Returns:
    boxes: Processed boxes in point form [0, 1], shape (n_frames, n_faces, 4)
    info: Processed info: idx, scanned, scan_found_face, confidence. Shape (n_frames, n_faces, 4)
  """
  # Make sure that enough frames are present
  if n_frames == 1:
    logging.warning("Ignoring enforce_consistency since n_frames=={}".format(n_frames))
    return boxes, info
  # Determine the maximum number of detections in any frame
  max_det_faces = int(np.max(np.sum(info[...,2], axis=-1)))
  # Trim to max_det_faces
  boxes = boxes[:,0:max_det_faces]
  info = info[:,0:max_det_faces]
  def distance_minimizing_idxs(boxes, info, max_det_faces):
    """Compute index permutations along second axis that minimize distances"""
    centers = (boxes[...,2:] + boxes[...,:2])/2
    valid = info[...,2]
    idx_perms = list(itertools.permutations(range(max_det_faces)))
    prev_c = centers[0]
    prev_idx_perm = idx_perms[0]
    idxs = []
    for c, v in zip(centers, valid):
      # Assume last known position for any non-valid faces.
      c[v == 0] = prev_c[v == 0]
      # Order according to prev_idx_perm
      c_ = np.take(c, prev_idx_perm, axis=0)
      # Compute distances at all possible permutations
      dists = [np.mean(np.linalg.norm(prev_c - np.take(c_, p, axis=0), axis=1)) for p in idx_perms]
      # Select permutation of idxs that minimizes distance to prev_centers
      idx_perm = idx_perms[np.argmin(dists)]
      idxs.append(idx_perm)
      prev_c, prev_idx_perm = c, idx_perm
    return np.asarray(idxs)[...,np.newaxis]
  # Make second axis temporally consistent by minimizing distances between centers
  idxs = distance_minimizing_idxs(boxes, info, max_det_faces)
  boxes = np.take_along_axis(boxes, idxs, axis=1)
  info = np.take_along_axis(info, idxs, axis=1)
  # Sort second axis by total confidence
  order = np.argsort(np.sum(info[...,3], axis=0))[::-1][np.newaxis]
  boxes = np.squeeze(np.take(boxes, order, axis=1), axis=1)
  info = np.squeeze(np.take(info, order, axis=1), axis=1)
  return boxes, info

def interpolate_unscanned_frames(
    boxes: np.ndarray,
    info: np.ndarray,
    n_frames: int
  ) -> Tuple[np.ndarray, np.ndarray]:
  """Interpolate values for frames that were not scanned.

  Args:
    boxes: Detected boxes in point form [0, 1], shape (n_frames, n_faces, 4)
    info: Detection info: idx, scanned, scan_found_face, interp_valid, confidence. Shape (n_frames, n_faces, 5)
    n_frames: Number of frames in the original input.
  Returns:
    boxes: Processed boxes in point form [0, 1], shape (orig_n_frames, n_faces, 4)
    info: Processed info: idx, scanned, scan_found_face, confidence. Shape (orig_n_frames, n_faces, 4)
  """
  _, n_faces, _ = info.shape
  # Add rows corresponding to unscanned frames
  add_idxs = list(set.difference(set(range(n_frames)), set(info[:,0,0].astype(np.int32).tolist())))
  idxs = np.repeat(np.asarray(add_idxs)[:,np.newaxis], n_faces, axis=1)[...,np.newaxis]
  # Info
  add_info = np.r_['2', idxs, np.zeros_like(idxs, np.int32), np.zeros_like(idxs, np.int32), np.zeros_like(idxs, np.int32)]
  info = np.concatenate([info, add_info])
  # Boxes
  add_boxes = np.full([len(add_idxs), n_faces, 4], np.nan)
  boxes = np.concatenate([boxes, add_boxes])
  # Sort according to frame idxs
  sort_idxs = np.argsort(info[:,0,0])
  boxes = np.take(boxes, sort_idxs, axis=0)
  info = np.take(info, sort_idxs, axis=0)
  # Interpolation
  boxes = np.apply_along_axis(interpolate_vals, 0, boxes)
  return boxes, info

class FaceDetector:
  def __init__(
      self,
      max_faces: int,
      fs: float,
      score_threshold: float,
      iou_threshold: float):
    """Initialise the face detector.

    Args:
      max_faces: The maximum number of faces to detect.
      fs: Frequency [Hz] at which faces should be detected. Detections are
        linearly interpolated for remaining frames.
      score_threshold: Face detection score threshold.
      iou_threshold: Face detection iou threshold.
    """
    import onnxruntime as rt
    with files('vitallens.models.Ultra-Light-Fast-Generic-Face-Detector-1MB') as model_dir:
      self.model = rt.InferenceSession(os.path.join(model_dir, "model_rfb_320.onnx"), providers=['CPUExecutionProvider'])
    self.iou_threshold = iou_threshold if iou_threshold is not None else self.config['threshold']
    self.score_threshold = score_threshold
    self.max_faces = max_faces
    self.fs = fs
  def __call__(
      self,
      inputs: Tuple[np.ndarray, str],
      inputs_shape: Tuple[tuple, float],
      fps: float
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference.

    Args:
      inputs: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        with a sequence of frames in unscaled uint8 RGB format, or a path to a video file.
      inputs_shape: The shape of the input video as (n_frames, h, w, 3)
      fps: Sampling frequency of the input video.
    Returns:
      boxes: Detected face boxes in relative flat point form (n_frames, n_faces, 4)
      info: Tuple (idx, scanned, scan_found_face, interp_valid, confidence) (n_frames, n_faces, 5)
    """
    # Determine number of batches
    n_frames = inputs_shape[0]
    n_batches = math.ceil((n_frames / (fps / self.fs)) / MAX_SCAN_FRAMES)
    if n_batches > 1:
      logging.info("Running face detection in {} batches...".format(n_batches))
    # Determine frame offsets for batches
    offsets_lengths = [(i[0], len(i)) for i in np.array_split(np.arange(n_frames), n_batches)]
    # Process in batches
    results = [self.scan_batch(inputs=inputs, batch=i, n_batches=n_batches, start=int(s), end=int(s+l), fps=fps) for i, (s, l) in enumerate(offsets_lengths)]
    boxes = np.concatenate([r[0] for r in results], axis=0)
    classes = np.concatenate([r[1] for r in results], axis=0)
    scan_idxs = np.concatenate([r[2] for r in results], axis=0)
    scan_every = int(np.max(np.diff(scan_idxs)))
    n_frames_scan = boxes.shape[0]
    # Non-max suppression
    idxs, num_valid = nms(boxes=boxes,
                          scores=classes[..., 1],
                          max_output_size=self.max_faces,
                          iou_threshold=self.iou_threshold,
                          score_threshold=self.score_threshold)
    max_valid = np.max(num_valid)
    idxs = idxs[...,0:max_valid]
    boxes = boxes[np.arange(boxes.shape[0])[:, None], idxs]
    classes = classes[np.arange(classes.shape[0])[:, None], idxs]
    # Check if any faces found
    if max_valid == 0:
      logging.warning("No faces found")
      return [], []
    # Assort info: idx, scanned, scan_found_face, confidence
    idxs = np.repeat(scan_idxs[:,np.newaxis], max_valid, axis=1)[...,np.newaxis]
    scanned = np.ones((n_frames_scan, max_valid, 1), dtype=np.int32)
    scan_found_face = np.where(classes[...,1:2] < self.score_threshold, np.zeros([n_frames_scan, max_valid, 1], dtype=np.int32), scanned)
    info = np.r_['2', idxs, scanned, scan_found_face, classes[...,1:2]]
    # Enforce temporal consistency
    boxes, info = enforce_temporal_consistency(boxes=boxes, info=info, n_frames=n_frames)
    # Interpolate unscanned frames if necessary
    if scan_every > 1:
      # Set unsuccessful detections to nan
      nan = info[:,:,2] == 0
      boxes[nan] = np.nan
      # Interpolate
      boxes, info = interpolate_unscanned_frames(boxes=boxes, info=info, n_frames=n_frames)
    # Return
    return boxes, info
  def scan_batch(
      self,
      batch: int,
      n_batches: int,
      inputs: Tuple[np.ndarray, str],
      start: int,
      end: int,
      fps: float = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Parse video and run inference for one batch.

    Args:
      batch: The number of this batch.
      n_batches: The total number of batches.
      inputs: The video to analyze. Either a np.ndarray of shape (n_frames, h, w, 3)
        with a sequence of frames in unscaled uint8 RGB format, or a path to a video file.
      start: The index of first frame of the video to analyze in this batch.
      end: The index of the last frame of the video to analyze in this batch.
      fps: Sampling frequency of the input video. Required if type(video) == np.ndarray.
    Returns:
      boxes: Scanned boxes in flat point form (n_frames, n_boxes, 4)
      classes: Detection scores for boxes (n_frames, n_boxes, 2)
      idxs: Indices of the scanned frames from the original video
    """
    logging.debug("Batch {}/{}...".format(batch, n_batches))
    # Parse the inputs
    inputs, fps, _, _, idxs = parse_video_inputs(
      video=inputs, fps=fps, target_size=INPUT_SIZE, target_fps=self.fs,
      library='prpy', scale_algorithm='bilinear', trim=(start, end))
    # Forward pass
    onnx_inputs = {"args_0": (inputs.astype(np.float32) - 127.0) / 128.0}
    onnx_outputs = self.model.run(None, onnx_inputs)[0]
    boxes = onnx_outputs[..., -4:]
    classes = onnx_outputs[..., 0:2]
    # Return
    return boxes, classes, idxs
    
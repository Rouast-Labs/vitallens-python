# Copyright (c) 2026 Rouast Labs
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

from dataclasses import dataclass
import logging
import numpy as np
from prpy.numpy.image import parse_image_inputs
import queue
import threading
import time
from typing import Callable, Optional
import uuid
import vitallens_core as vc

from vitallens.methods.simple_rppg_method import SimpleRPPGMethod

@dataclass
class InferenceContext:
  timestamp: float
  roi: vc.Rect
  face_conf: float

class FrameBuffer:
  def __init__(self, buffer_id: str, roi: vc.Rect, max_capacity: int, timestamp: float):
    """Initialize a thread-safe frame buffer for a specific region of interest.

    Args:
      buffer_id: Unique identifier for the buffer.
      roi: The region of interest (ROI) associated with this buffer.
      max_capacity: The maximum number of frames the buffer can hold.
      timestamp: The creation timestamp of the buffer.
    """
    self.id = buffer_id
    self.roi = roi
    self.created_at = timestamp
    self.last_seen = timestamp
    self.max_capacity = max_capacity
    self.buffer = []

  @property
  def count(self) -> int:
    return len(self.buffer)

  def append(self, frame: np.ndarray, context: InferenceContext):
    """Add a frame and its metadata to the buffer, maintaining max capacity.

    Args:
      frame: The parsed RGB image data.
      context: Metadata including timestamp and ROI.
    """
    self.buffer.append((frame, context))
    self.last_seen = context.timestamp
    if len(self.buffer) > self.max_capacity:
      overflow = len(self.buffer) - self.max_capacity
      self.buffer = self.buffer[overflow:]

  def execute(self, take_count: int, keep_count: int) -> list:
    """Extract frames for inference and manage buffer overlap.

    Args:
      take_count: Number of frames to extract for the current batch.
      keep_count: Number of frames to retain for the next sliding window.
    Returns:
      payload: A list of (frame, context) tuples, or None if insufficient frames.
    """
    if take_count <= 0 or len(self.buffer) < take_count:
      return None
    payload = self.buffer[:take_count]
    elements_to_remove = max(0, take_count - keep_count)
    self.buffer = self.buffer[elements_to_remove:]
    return payload\

class BufferManager:
  """Thread-safe manager for multiple streaming frame buffers."""
  def __init__(self, buffer_config: vc.BufferConfig):
    """Initialize the manager with specific buffering and overlap constraints.

    Args:
      buffer_config: Configuration for stream window sizes and overlaps.
    """
    self.buffer_planner = vc.BufferPlanner(buffer_config)
    self.buffer_config = buffer_config
    self.buffers = {}
    self.state = None
    self.current_timestamp = 0.0
    self.lock = threading.Lock()

  def _get_active_metadata(self):
    return [
      vc.BufferMetadata(
        id=buf.id, roi=buf.roi, count=buf.count, 
        created_at=buf.created_at, last_seen=buf.last_seen
      ) for buf in self.buffers.values()
    ]

  def register_target(self, target_roi: vc.Rect, timestamp: float):
    """Register or update a target ROI in the buffer planner.

    Args:
      target_roi: The latest detected face ROI to track.
      timestamp: The current frame timestamp.
    Returns:
      buffer_id: The ID of the buffer assigned to this target, or None.
    """
    with self.lock:
      self.current_timestamp = max(self.current_timestamp, timestamp)
      action = self.buffer_planner.evaluate_target(target_roi, timestamp, self._get_active_metadata())
      if action.action == vc.BufferActionType.Create:
        new_id = str(uuid.uuid4())
        roi = action.roi if action.roi is not None else target_roi
        max_cap = self.buffer_config.stream_max + 50 
        self.buffers[new_id] = FrameBuffer(new_id, roi, max_cap, timestamp)
        return new_id
      elif action.action == vc.BufferActionType.KeepAlive:
        if action.matched_id in self.buffers:
          self.buffers[action.matched_id].last_seen = timestamp
        return action.matched_id
      return None

  def append(self, buffer_id: str, frame: np.ndarray, context: InferenceContext):
    """Route a frame to a specific active buffer.

    Args:
      buffer_id: The identifier for the target buffer.
      frame: The parsed RGB image data.
      context: Metadata including timestamp and ROI.
    """
    with self.lock:
      self.current_timestamp = max(self.current_timestamp, context.timestamp)
      if buffer_id in self.buffers:
        self.buffers[buffer_id].append(frame, context)

  def poll(self, flush: bool = False):
    """Poll the planner for the next required inference command.

    Args:
      flush: If True, forces the planner to return commands for remaining data.
    Returns:
      command: A vc.InferenceCommand containing the buffer ID and frame counts.
    """
    with self.lock:
      has_state = self.state is not None
      plan = self.buffer_planner.poll(
        self._get_active_metadata(), 
        self.current_timestamp, 
        vc.InferenceMode.Stream, 
        has_state, 
        flush
      )
      for drop_id in plan.buffers_to_drop:
        if drop_id in self.buffers:
          del self.buffers[drop_id]
      return plan.command

  def execute(self, command: vc.InferenceCommand) -> list:
    """Execute a specific inference command by extracting data from the relevant buffer.

    Args:
      command: The command containing buffer ID and frame counts.
    Returns:
      window: A list of frames and contexts ready for inference.
    """
    with self.lock:
      if command.buffer_id in self.buffers:
        return self.buffers[command.buffer_id].execute(command.take_count, command.keep_count)
      return None

  def get_state(self):
    """Retrieve the current physiological state of the rPPG model.

    Returns:
      state: The model's internal state vector or None.
    """
    with self.lock:
      return self.state

  def update_state(self, new_state):
    """Update the internal state of the rPPG model after an inference step.

    Args:
      new_state: The updated state vector returned by the API.
    """
    with self.lock:
      self.state = new_state

  def get_all_buffers(self):
    """Get a list of all currently active buffers.

    Returns:
      buffers: A list of (buffer_id, roi) tuples.
    """
    with self.lock:
      return [(buf.id, buf.roi) for buf in self.buffers.values()]

  def reset(self):
    """Clear all active buffers and reset the model state."""
    with self.lock:
      self.buffers.clear()
      self.state = None

class StreamSession:
  """Context manager handling background inference and buffer synchronization for live streams."""
  def __init__(self, rppg_method, face_detector=None, fdet_fs=1.0, on_result: Optional[Callable] = None):
    """Initialize the streaming session and start the background inference thread.

    Args:
      rppg_method: The method instance (e.g., VitalLensRPPGMethod) to use.
      face_detector: Optional instance of FaceDetector for automatic tracking.
      fdet_fs: Frequency [Hz] at which to run automatic face detection.
      on_result: Optional callback for asynchronous result handling.
    """
    self.rppg = rppg_method
    self.face_detector = face_detector
    self.fdet_fs = fdet_fs
    self.on_result = on_result
    self.result_queue = queue.Queue()
    self.buffer_manager = BufferManager(vc.compute_buffer_config(self.rppg.session_config))
    self.vc_session = vc.Session(self.rppg.session_config)
    self.last_fdet_time = -1.0
    self.last_processed_time = -1.0
    self.current_face = None
    self.current_roi_rust = None
    self.running = True
    self.thread = threading.Thread(target=self._inference_loop, daemon=True)
    self.thread.start()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.close()

  def close(self):
    """Stop the background thread and shut down the streaming session."""
    self.running = False
    if self.thread.is_alive():
      self.thread.join(timeout=2.0)

  def push(self, frame: np.ndarray, timestamp: float, face: np.ndarray = None):
    """Push a single frame and its timestamp into the streaming pipeline.

    Args:
      frame: The input video frame of shape (h, w, 3) in RGB format.
      timestamp: The absolute timestamp of the frame in seconds.
      face: Optional pre-detected face box [x0, y0, x1, y1].
    """
    if not self.running: return
    # Throttled face detection
    min_interval = 1.0 / self.rppg.fps_target
    if (timestamp - self.last_processed_time) < (min_interval - 0.005):
      return
    self.last_processed_time = timestamp
    if face is not None:
      self.current_face = face
    elif self.face_detector and (timestamp - self.last_fdet_time) >= (1.0 / self.fdet_fs):
      faces_rel, _ = self.face_detector(inputs=frame[np.newaxis], n_frames=1, fps=30.0)
      if len(faces_rel) > 0 and len(faces_rel[0]) > 0:
        h, w = frame.shape[:2]
        self.current_face = (faces_rel[0][0] * [w, h, w, h]).astype(np.int64)
      self.last_fdet_time = timestamp
    if self.current_face is None:
      return
    # ROI calculation and registration
    if self.current_roi_rust is None or (timestamp - self.last_fdet_time) < 0.05:
      self.current_roi_rust = vc.calculate_roi(
        face=vc.Rect(
          x=float(self.current_face[0]),
          y=float(self.current_face[1]),
          width=float(self.current_face[2] - self.current_face[0]),
          height=float(self.current_face[3] - self.current_face[1])
        ),
        method=self.rppg.roi_method,
        container=(float(frame.shape[1]), float(frame.shape[0]))
      )
    self.buffer_manager.register_target(self.current_roi_rust, timestamp)

    active_buffers = self.buffer_manager.get_all_buffers()
    if not active_buffers:
      return

    # Process the frame for active buffer using its specific ROI
    for buf_id, buf_roi in active_buffers:
      # Clamp ROI to image boundaries
      x0 = max(0, int(buf_roi.x))
      y0 = max(0, int(buf_roi.y))
      x1 = min(frame.shape[1], int(buf_roi.x + buf_roi.width))
      y1 = min(frame.shape[0], int(buf_roi.y + buf_roi.height))
      
      if x1 <= x0 or y1 <= y0:
        continue

      # Save to parsed_frame so we don't overwrite the original frame
      parsed_frame, _, _, _, _ = parse_image_inputs(
        inputs=frame, fps=30.0,
        roi=(x0, y0, x1, y1),
        target_size=self.rppg.input_size, preserve_aspect_ratio=False,
        library='prpy', scale_algorithm='bilinear', allow_image=True, videodims=False
      )

      if isinstance(self.rppg, SimpleRPPGMethod):
        payload = np.mean(parsed_frame, axis=(0, 1))
      else:
        payload = parsed_frame

      ctx = InferenceContext(timestamp=timestamp, roi=buf_roi, face_conf=1.0)
      self.buffer_manager.append(buf_id, payload, ctx)

  def get_result(self, block=False, timeout=None):
    """Pull the latest inference results from the session queue.

    Args:
      block: Whether to block until a result is available.
      timeout: Maximum time to block if block is True.
    Returns:
      results: A list of analysis results for detected faces.
    """
    try:
      return self.result_queue.get(block=block, timeout=timeout)
    except queue.Empty:
      return None

  def _inference_loop(self):
    """Background loop that polls for ready buffers and executes model inference."""
    while self.running:
      command = self.buffer_manager.poll()
      if not command:
        time.sleep(0.01)
        continue
      window = self.buffer_manager.execute(command)
      if not window:
        continue
      window_frames = np.stack([item[0] for item in window])
      window_faces = np.stack([[item[1].roi.x, item[1].roi.y, item[1].roi.x+item[1].roi.width, item[1].roi.y+item[1].roi.height] for item in window])
      # Dynamically compute fps from timestamps
      timestamps = [item[1].timestamp for item in window]
      actual_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else self.rppg.fps_target
      # Execute strategy
      try:
        sig, conf, live, new_state = self.rppg.infer_stream(
          window_frames, actual_fps, self.buffer_manager.get_state()
        )
        self.buffer_manager.update_state(new_state)
        n_res = len(live)
        timestamps = timestamps[-n_res:]
        window_faces = window_faces[-n_res:]
        signals_input = {k: vc.SignalInput(data=v.tolist(), confidence=conf[k].tolist()) for k, v in sig.items()}
        face_input = vc.FaceInput(coordinates=window_faces.tolist(), confidence=live.tolist())
        session_input = vc.SessionInput(face=face_input, signals=signals_input, timestamp=timestamps)
        session_result = self.vc_session.process(session_input, "Incremental")
        res_dict = self._format_result(session_result)
        if self.on_result:
          self.on_result(res_dict)
        self.result_queue.put(res_dict)
      except Exception as e:
        logging.error(f"Stream inference error: {e}")
        self.buffer_manager.reset()
        self.vc_session.reset()

  def _format_result(self, session_result):
    """Format the raw session result into the standard vitallens-python result dictionary.

    Args:
      session_result: The vc.SessionResult object from vitallens-core.
    Returns:
      res_dict: A list containing a formatted results dictionary for the face.
    """
    face_dict = {
      'coordinates': [], 
      'confidence': [], 
      'note': 'Face detection coordinates for this face with live confidence levels.'
    }
    
    if session_result.face is not None:
      face_dict['coordinates'] = session_result.face.coordinates
      face_dict['confidence'] = session_result.face.confidence
      if session_result.face.note:
        face_dict['note'] = session_result.face.note

    res_dict = {
      'face': face_dict,
      'vitals': {},
      'waveforms': {},
      'message': session_result.message,
      'fps': session_result.fps,
      'n': len(session_result.timestamp),
      'time': session_result.timestamp
    }

    for key, wave in session_result.waveforms.items():
      res_dict['waveforms'][key] = {
        'data': wave.data,
        'unit': wave.unit,
        'confidence': wave.confidence,
        'note': wave.note
      }

    for key, vital in session_result.vitals.items():
      res_dict['vitals'][key] = {
        'value': vital.value,
        'unit': vital.unit,
        'confidence': vital.confidence,
        'note': vital.note
      }

    return [res_dict]
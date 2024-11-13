import argparse
import concurrent.futures
import cv2
import numpy as np
from prpy.numpy.face import get_upper_body_roi_from_det
from prpy.numpy.signal import estimate_freq
import sys
import threading
import time
import warnings

sys.path.append('../vitallens-python')
from vitallens import VitalLens, Mode, Method
from vitallens.buffer import SignalBuffer, MultiSignalBuffer
from vitallens.constants import API_MIN_FRAMES

def draw_roi(frame, roi):
  roi = np.asarray(roi).astype(np.int32)
  frame = cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 1)

def draw_signal(frame, roi, sig, sig_name, sig_conf_name, draw_area_tl_x, draw_area_tl_y, color):
  def _draw(frame, vals, display_height, display_width, min_val, max_val, color, thickness):
    height_mult = display_height/(max_val - min_val)
    width_mult = display_width/(vals.shape[0] - 1)
    p1 = (int(draw_area_tl_x), int(draw_area_tl_y + (max_val - vals[0]) * height_mult))
    for i, s in zip(range(1, len(vals)), vals[1:]):
      p2 = (int(draw_area_tl_x + i * width_mult), int(draw_area_tl_y + (max_val - s) * height_mult))
      frame = cv2.line(frame, p1, p2, color, thickness)
      p1 = p2
  # Derive dims from roi
  display_height = (roi[3] - roi[1]) / 2.0
  display_width = (roi[2] - roi[0]) * 0.8
  # Draw signal
  if sig_name in sig:
    vals = np.asarray(sig[sig_name])
    min_val = np.min(vals)
    max_val = np.max(vals)
    if max_val - min_val == 0:
      return frame
    _draw(frame=frame, vals=vals, display_height=display_height, display_width=display_width,
          min_val=min_val, max_val=max_val, color=color, thickness=2)
  # Draw confidence
  if sig_conf_name in sig:
    vals = np.asarray(sig[sig_conf_name])
    _draw(frame=frame, vals=vals, display_height=display_height, display_width=display_width,
          min_val=0., max_val=1., color=color, thickness=1)

def draw_fps(frame, fps, text, draw_area_bl_x, draw_area_bl_y):
  cv2.putText(frame, text='{}: {:.1f}'.format(text, fps), org=(draw_area_bl_x, draw_area_bl_y),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,255,0), thickness=1)

def draw_vital(frame, sig, text, sig_name, fps, mult, color, draw_area_bl_x, draw_area_bl_y):
  if sig_name in sig:
    val = estimate_freq(x=sig[sig_name], f_s=fps, f_res=0.0167, method='periodogram') * mult
    cv2.putText(frame, text='{}: {:.1f}'.format(text, val), org=(draw_area_bl_x, draw_area_bl_y),
      fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=color, thickness=1)

class VitalLensRunnable:
  def __init__(self, method, api_key):
    self.active = threading.Event()
    self.result = []
    self.vl = VitalLens(method=method,
                        mode=Mode.BURST,
                        api_key=api_key,
                        detect_faces=True,
                        estimate_running_vitals=True,
                        export_to_json=False)
  def __call__(self, inputs, fps):
    self.active.set()
    self.result = self.vl(np.asarray(inputs), fps=fps)
    self.active.clear()

def run(args):
  cap = cv2.VideoCapture(0)
  executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
  vl = VitalLensRunnable(method=args.method, api_key=args.api_key)
  signal_buffer = MultiSignalBuffer(size=120, ndim=1, ignore_k=['face'])
  fps_buffer = SignalBuffer(size=120, ndim=1, pad_val=np.nan)
  frame_buffer = []
  # Sample frames from cv2 video stream attempting to achieve this framerate
  target_fps = 30.
  # Check if the webcam is opened correctly
  if not cap.isOpened():
    raise IOError("Cannot open webcam")
  # Read first frame to get dims
  _, frame = cap.read()
  height, width, _ = frame.shape
  roi = None
  i = 0
  t, p_t = time.time(), time.time()
  fps, p_fps = 30.0, 30.0
  ds_factor = 1
  n_frames = 0
  signals = None
  while True:
    ret, frame = cap.read()
    # Measure frequency
    t_prev = t
    t = time.time()
    if not vl.active.is_set():
      # Process result if available
      if len(vl.result) > 0:
        # Results are available - fetch and reset
        result = vl.result[0]
        vl.result = []
        # Update the buffer
        signals = signal_buffer.update({
          **{
            f"{key}_sig": value['value'] if 'value' in value else np.array(value['data'])
            for key, value in result['vital_signs'].items()
          },
          **{
            f"{key}_conf": value['confidence'] if isinstance(value['confidence'], np.ndarray) else np.array(value['confidence'])
            for key, value in result['vital_signs'].items()
          },
          'face_conf': result['face']['confidence'],
        }, dt=n_frames)
        with warnings.catch_warnings():
          warnings.simplefilter("ignore", category=RuntimeWarning)
          # Measure actual effective sampling frequency at which neural net input was sampled
          fps = np.nanmean(fps_buffer.update([(1./(t - t_prev))/ds_factor], dt=n_frames))
        roi = get_upper_body_roi_from_det(result['face']['coordinates'][-1], clip_dims=(width, height), cropped=True)
        # Measure prediction frequency - how often predictions are made
        p_t_prev = p_t
        p_t = time.time()
        p_fps = 1./(p_t - p_t_prev)
      else:
        # No results available
        roi = None
        signal_buffer.clear()
      # Start next prediction
      if len(frame_buffer) >= (API_MIN_FRAMES if args.method == Method.VITALLENS else 1):
        n_frames = len(frame_buffer)
        future = executor.submit(vl, frame_buffer.copy(), fps)
        frame_buffer.clear()
    # Sample frames
    if i % ds_factor == 0:
      # Add current frame to the buffer (BGR -> RGB)
      frame_buffer.append(frame[...,::-1])
    i += 1
    # Display
    if roi is not None:
      draw_roi(frame, roi)
      draw_signal(
        frame=frame, roi=roi, sig=signals, sig_name='ppg_waveform_sig', sig_conf_name='ppg_waveform_conf',
        draw_area_tl_x=roi[2]+20, draw_area_tl_y=roi[1], color=(0, 0, 255))
      draw_signal(
        frame=frame, roi=roi, sig=signals, sig_name='respiratory_waveform_sig', sig_conf_name='respiratory_waveform_conf',
        draw_area_tl_x=roi[2]+20, draw_area_tl_y=int(roi[1]+(roi[3]-roi[1])/2.0), color=(255, 0, 0))
      draw_fps(frame, fps=fps, text="fps", draw_area_bl_x=roi[0], draw_area_bl_y=roi[3]+20)
      draw_fps(frame, fps=p_fps, text="p_fps", draw_area_bl_x=int(roi[0]+0.4*(roi[2]-roi[0])), draw_area_bl_y=roi[3]+20)
      draw_vital(frame, sig=signals, text="hr [bpm]", sig_name='ppg_waveform_sig', fps=fps, mult=60., color=(0,0,255), draw_area_bl_x=roi[2]+20, draw_area_bl_y=int(roi[1]+(roi[3]-roi[1])/2.0))
      draw_vital(frame, sig=signals, text="rr [rpm]", sig_name='respiratory_waveform_sig', fps=fps, mult=60., color=(255,0,0), draw_area_bl_x=roi[2]+20, draw_area_bl_y=roi[3])
    cv2.imshow('Live', frame)
    c = cv2.waitKey(1)
    if c == 27:
      break
    # Even out fps
    dt_req = 1./target_fps - (time.time() - t)
    if dt_req > 0: time.sleep(dt_req)

  cap.release()
  cv2.destroyAllWindows()

def method_type(name):
  try:
    return Method[name]
  except KeyError:
    raise argparse.ArgumentTypeError(f"{name} is not a valid Method")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--api_key', type=str, default='', help='Your API key. Get one for free at https://www.rouast.com/api.')
  parser.add_argument('--method', type=method_type, default='VITALLENS', help='Choice of method (VITALLENS, POS, CHROM, or G)')
  args = parser.parse_args()
  run(args)

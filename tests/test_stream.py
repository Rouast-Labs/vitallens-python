import numpy as np
import time
import vitallens_core as vc
from vitallens.stream import FrameBuffer, InferenceContext, BufferManager, StreamSession
from vitallens.methods.simple_rppg_method import SimpleRPPGMethod

class DummyStreamRPPG(SimpleRPPGMethod):
  def __init__(self):
    super().__init__()
    config = vc.SessionConfig(
        model_name="dummy",
        supported_vitals=["heart_rate"],
        return_waveforms=["ppg_waveform"],
        fps_target=30.0,
        input_size=10,
        n_inputs=1,
        roi_method="face"
    )
    self.parse_config(config, est_window_length=5, est_window_overlap=0)
    self.input_size = config.input_size

  def algorithm(self, rgb: np.ndarray, fps: float) -> np.ndarray:
    return np.mean(rgb, axis=-1)

  def pulse_filter(self, sig: np.ndarray, fps: float) -> np.ndarray:
    return sig

def test_frame_buffer():
  roi = vc.Rect(x=0.0, y=0.0, width=10.0, height=10.0)
  buf = FrameBuffer(buffer_id="test_id", roi=roi, max_capacity=5, timestamp=1.0)
  for i in range(7):
    ctx = InferenceContext(timestamp=float(i), roi=roi, face_conf=1.0)
    buf.append(np.zeros((10, 10, 3)), ctx)
  assert buf.count == 5
  assert buf.last_seen == 6.0
  payload = buf.execute(take_count=3, keep_count=1)
  assert len(payload) == 3
  assert buf.count == 3

def test_buffer_manager():
  config = vc.SessionConfig(
    model_name="dummy",
    supported_vitals=[],
    fps_target=30.0,
    input_size=10,
    n_inputs=4,
    roi_method="face"
  )
  buf_config = vc.compute_buffer_config(config)
  manager = BufferManager(buf_config)
  roi = vc.Rect(x=0.0, y=0.0, width=10.0, height=10.0)
  buf_id = manager.register_target(roi, 1.0)
  assert buf_id is not None
  for i in range(16):
    ctx = InferenceContext(timestamp=float(i)/30.0, roi=roi, face_conf=1.0)
    manager.append(buf_id, np.zeros((10, 10, 3)), ctx)
  command = manager.poll()
  assert command is not None
  assert command.buffer_id == buf_id
  payload = manager.execute(command)
  assert payload is not None
  assert len(payload) == command.take_count

def test_stream_session():
  rppg = DummyStreamRPPG()
  with StreamSession(rppg_method=rppg, face_detector=None) as session:
    for i in range(25):
      frame = np.zeros((480, 640, 3), dtype=np.uint8)
      face = np.array([100, 100, 200, 200])
      session.push(frame, timestamp=float(i)/30.0, face=face)
    time.sleep(0.5)
    res = session.get_result(block=False)
    assert res is not None
    assert len(res) > 0
    assert 'vitals' in res[0]
    assert 'waveforms' in res[0]
    assert 'ppg_waveform' in res[0]['waveforms']
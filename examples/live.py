import argparse
from collections import deque
import cv2
import numpy as np
import time
from vitallens import VitalLens
import vitallens_core as vc

def hex_to_bgr(hex_color):
  if not hex_color: return (255, 255, 255)
  hex_color = hex_color.lstrip('#')
  r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
  return (b, g, r)

def draw_waveform(frame, data, color, rect, title):
  x, y, w, h = rect
  cv2.putText(frame, title, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
  if len(data) < 2: return

  min_val, max_val = min(data), max(data)
  if max_val - min_val == 0: return

  pts = []
  step = w / (len(data) - 1)
  for i, val in enumerate(data):
      px = int(x + i * step)
      py = int(y + h - ((val - min_val) / (max_val - min_val)) * h)
      pts.append((px, py))

  pts = np.array(pts, np.int32).reshape((-1, 1, 2))
  cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

def main():
  parser = argparse.ArgumentParser(description="VitalLens Live Webcam Demo")
  parser.add_argument('--method', type=str, default='pos', help='Method to use (e.g., pos, chrom, g, vitallens)')
  parser.add_argument('--api_key', type=str, default=None, help='API key (required for vitallens method)')
  args = parser.parse_args()

  vl = VitalLens(method=args.method, api_key=args.api_key)
  vl.rppg.fps_target = 15.0

  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("Error: Could not open webcam.")
    return

  print(f"Starting live stream using {args.method}. Press 'q' to quit.")

  font = cv2.FONT_HERSHEY_SIMPLEX
  latest_vitals = {}
  latest_coords = None
  ppg_history = deque(maxlen=150)
  ppg_conf = deque(maxlen=150)
  resp_history = deque(maxlen=150)
  resp_conf = deque(maxlen=150)
  vital_conf_thresh = 0.8
  hrv_conf_thresh = 0.7
  start_time = time.time()

  ppg_meta = vc.get_vital_info("ppg_waveform")
  hr_meta = vc.get_vital_info("heart_rate")
  sdnn_meta = vc.get_vital_info("hrv_sdnn")
  rmssd_meta = vc.get_vital_info("hrv_rmssd")
  resp_meta = vc.get_vital_info("respiratory_waveform")
  rr_meta = vc.get_vital_info("respiratory_rate")

  with vl.stream() as session:
    while True:
      ret, frame = cap.read()
      if not ret:
        break

      timestamp = time.time() - start_time
      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

      session.push(rgb_frame, timestamp)

      if session.current_face is not None:
        if latest_coords is None:
          print("[DEBUG] Face initially detected!")
        latest_coords = session.current_face

      res = session.get_result(block=False)

      if res and len(res) > 0:
        new_vitals = res[0].get('vitals', {})
        if new_vitals:
          latest_vitals.update(new_vitals)

        waveforms = res[0].get('waveforms', {})
        if 'ppg_waveform' in waveforms:
          ppg_history.extend(waveforms['ppg_waveform']['data'])
          ppg_conf.extend(waveforms['ppg_waveform']['confidence'])
        if 'respiratory_waveform' in waveforms:
          resp_history.extend(waveforms['respiratory_waveform']['data'])
          resp_conf.extend(waveforms['respiratory_waveform']['confidence'])

      if session.current_face is None:
        state_text = "Searching"
        msg_text = "Position your face in the frame"
        color = (0, 165, 255)
      elif 'heart_rate' not in latest_vitals:
        state_text = "Calibrating"
        msg_text = "Hold still and ensure good lighting..."
        color = (255, 0, 255)
      else:
        state_text = "Tracking"
        msg_text = "Tracking vitals"
        color = (0, 255, 0)

      fh, fw = frame.shape[:2]

      # Top bar
      cv2.rectangle(frame, (0, 0), (fw, 55), (30, 30, 30), -1)
      cv2.putText(frame, state_text, (20, 25), font, 0.7, color, 2)
      cv2.putText(frame, msg_text, (20, 45), font, 0.5, (200, 200, 200), 1)

      if session.current_face is not None:
        x1, y1, x2, y2 = map(int, session.current_face)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

      # Bottom panel
      panel_h = 240
      cv2.rectangle(frame, (0, fh - panel_h), (fw, fh), (30, 30, 30), -1)

      n_samples = int(vl.rppg.fps_target)
      ppg_last_sec = list(ppg_conf)[-n_samples:]
      resp_last_sec = list(resp_conf)[-n_samples:]
      avg_ppg_conf = sum(ppg_last_sec) / len(ppg_last_sec) if ppg_last_sec else 0
      avg_resp_conf = sum(resp_last_sec) / len(resp_last_sec) if resp_last_sec else 0

      wave_w = fw - 240

      # Fetch metadata
      ppg_meta = vc.get_vital_info("ppg_waveform")
      hr_meta = vc.get_vital_info("heart_rate")
      sdnn_meta = vc.get_vital_info("hrv_sdnn")
      rmssd_meta = vc.get_vital_info("hrv_rmssd")
      resp_meta = vc.get_vital_info("respiratory_waveform")
      rr_meta = vc.get_vital_info("respiratory_rate")

      # --- PPG Row ---
      row1_y = fh - panel_h + 30
      if avg_ppg_conf >= vital_conf_thresh:
        draw_waveform(frame, list(ppg_history), hex_to_bgr(ppg_meta.color), (20, row1_y, wave_w - 40, 60), ppg_meta.display_name)

      # HR
      hr_val = "--"
      if 'heart_rate' in latest_vitals and latest_vitals['heart_rate']['confidence'] >= vital_conf_thresh:
        hr_val = f"{latest_vitals['heart_rate']['value']:.0f}"
      cv2.putText(frame, f"{hr_meta.short_name}", (wave_w, row1_y), font, 0.5, (200, 200, 200), 1)
      cv2.putText(frame, f"{hr_val} {hr_meta.unit}", (wave_w, row1_y + 30), font, 0.7, hex_to_bgr(hr_meta.color), 2)

      # SDNN
      sdnn_val = "--"
      if 'hrv_sdnn' in latest_vitals and latest_vitals['hrv_sdnn']['confidence'] >= hrv_conf_thresh:
        sdnn_val = f"{latest_vitals['hrv_sdnn']['value']:.0f}"
      cv2.putText(frame, f"{sdnn_meta.short_name}", (wave_w + 120, row1_y), font, 0.4, (200, 200, 200), 1)
      cv2.putText(frame, f"{sdnn_val} {sdnn_meta.unit}", (wave_w + 120, row1_y + 20), font, 0.5, hex_to_bgr(sdnn_meta.color), 1)

      # RMSSD
      rmssd_val = "--"
      if 'hrv_rmssd' in latest_vitals and latest_vitals['hrv_rmssd']['confidence'] >= hrv_conf_thresh:
        rmssd_val = f"{latest_vitals['hrv_rmssd']['value']:.0f}"
      cv2.putText(frame, f"{rmssd_meta.short_name}", (wave_w + 120, row1_y + 50), font, 0.4, (200, 200, 200), 1)
      cv2.putText(frame, f"{rmssd_val} {rmssd_meta.unit}", (wave_w + 120, row1_y + 70), font, 0.5, hex_to_bgr(rmssd_meta.color), 1)

      # --- Resp Row ---
      row2_y = fh - panel_h + 140
      if avg_resp_conf >= vital_conf_thresh:
        draw_waveform(frame, list(resp_history), hex_to_bgr(resp_meta.color), (20, row2_y, wave_w - 40, 60), resp_meta.display_name)

      rr_val = "--"
      if 'respiratory_rate' in latest_vitals and latest_vitals['respiratory_rate']['confidence'] >= vital_conf_thresh:
        rr_val = f"{latest_vitals['respiratory_rate']['value']:.0f}"
      cv2.putText(frame, f"{rr_meta.short_name}", (wave_w, row2_y), font, 0.5, (200, 200, 200), 1)
      cv2.putText(frame, f"{rr_val} {rr_meta.unit}", (wave_w, row2_y + 30), font, 0.7, hex_to_bgr(rr_meta.color), 2)

      cv2.imshow("VitalLens Live", frame)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
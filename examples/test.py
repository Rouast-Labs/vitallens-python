import sys
sys.path.append('../vitallens-python')
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
from prpy.ffmpeg.probe import probe_video
from prpy.ffmpeg.readwrite import read_video_from_path
from prpy.helpers import str2bool
from prpy.numpy.physio import estimate_hr_from_signal, estimate_rr_from_signal
from prpy.numpy.physio import EScope, EMethod
import timeit
from vitallens import VitalLens, Method
from vitallens.utils import download_file

COLOR_GT = '#000000'
METHOD_COLORS = {
  Method.VITALLENS: '#00a4df',
  Method.VITALLENS_1_0: '#00a4df',
  Method.VITALLENS_1_1: '#00a4df',
  Method.VITALLENS_2_0: '#00a4df',
  Method.G: '#00ff00',
  Method.CHROM: '#4ceaff',
  Method.POS: '#23b031'
}
SAMPLE_VIDEO_URLS = {
  'examples/sample_video_1.mp4': 'https://github.com/Rouast-Labs/vitallens-python/raw/main/examples/sample_video_1.mp4',
  'examples/sample_video_2.mp4': 'https://github.com/Rouast-Labs/vitallens-python/raw/main/examples/sample_video_2.mp4',
}
SAMPLE_VITALS_URLS = {
  'examples/sample_vitals_1.csv': 'https://github.com/Rouast-Labs/vitallens-python/raw/main/examples/sample_vitals_1.csv',
  'examples/sample_vitals_2.csv': 'https://github.com/Rouast-Labs/vitallens-python/raw/main/examples/sample_vitals_2.csv',
}

def run(args=None):
  # Download sample data if necessary
  if args.video_path in SAMPLE_VIDEO_URLS.keys() and not os.path.exists(args.video_path):
    download_file(url=SAMPLE_VIDEO_URLS[args.video_path], dest=args.video_path)
  if args.vitals_path in SAMPLE_VITALS_URLS.keys() and not os.path.exists(args.vitals_path):
    download_file(url=SAMPLE_VITALS_URLS[args.vitals_path], dest=args.vitals_path)
  # Get ground truth vitals
  vitals = pd.read_csv(args.vitals_path) if args.vitals_path and os.path.exists(args.vitals_path) else pd.DataFrame()
  ppg_gt = vitals['ppg'] if 'ppg' in vitals else None
  resp_gt = vitals['resp'] if 'resp' in vitals else None
  # Get video
  fps, *_ = probe_video(args.video_path)
  if args.input_str:
    video = args.video_path
    print(f"Using video at: {args.video_path}")
  else:
    print(f"Reading full video into memory from {args.video_path}...")
    video, _ = read_video_from_path(path=args.video_path, pix_fmt='rgb24')
    print(f"Video shape: {video.shape}")
  # Estimate vitals and measure inference time
  vl = VitalLens(method=args.method, api_key=args.api_key)
  start = timeit.default_timer()
  result = vl(video=video, fps=fps)
  stop = timeit.default_timer()
  time_ms = (stop - start) * 1000
  print(f"Inference time: {time_ms:.2f} ms")
  # Print the results
  print(result)
  # Plot the results
  if not result:
    print("No faces detected, cannot plot results.")
    return
  vital_signs = result[0]['vital_signs']
  if "respiratory_waveform" in vital_signs:
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(15, 8))
  else:
    fig, ax1 = plt.subplots(1, figsize=(15, 6))
  fig.suptitle(f"Vital signs from {args.video_path} using {args.method.name} ({time_ms:.2f} ms)")
  # PPG Waveform and Heart Rate Plot (ax1)
  ax1.set_ylabel('Waveform (unitless)', color=METHOD_COLORS[args.method])
  ax1.tick_params(axis='y', labelcolor=METHOD_COLORS[args.method])
  if "ppg_waveform" in vital_signs:
    hr_string = f" -> Global HR: {vital_signs['heart_rate']['value']:.1f} bpm" if "heart_rate" in vital_signs else ""
    ax1.plot(vital_signs['ppg_waveform']['data'], color=METHOD_COLORS[args.method], label=f"PPG Waveform{hr_string}", zorder=10)
    ax1.plot(vital_signs['ppg_waveform']['confidence'], color=METHOD_COLORS[args.method], linestyle='--', label='PPG Confidence', zorder=5)
  if ppg_gt is not None:
    hr_gt = estimate_hr_from_signal(signal=ppg_gt, f_s=fps, scope=EScope.GLOBAL, method=EMethod.PERIODOGRAM)
    ax1.plot(ppg_gt, color=COLOR_GT, label=f"Ground Truth PPG -> HR: {hr_gt:.1f} bpm", zorder=0)
  ax1_handles, ax1_labels = ax1.get_legend_handles_labels()
  if "rolling_heart_rate" in vital_signs:
    ax1_hr = ax1.twinx()
    hr_color = '#ff7f0e'
    ax1_hr.set_ylabel('Heart Rate (bpm)', color=hr_color)
    ax1_hr.tick_params(axis='y', labelcolor=hr_color)
    ax1_hr.set_ylim(35, 180)
    rolling_hr_data = vital_signs['rolling_heart_rate']['data']
    line_hr, = ax1_hr.plot(rolling_hr_data, color=hr_color, linestyle='-', label='Rolling Heart Rate')
    ax1_handles.append(line_hr)
    ax1_labels.append('Rolling Heart Rate')
  ax1.legend(ax1_handles, ax1_labels, loc='upper left')
  ax1.grid(True, linestyle=':', alpha=0.6)
  # Respiratory Waveform and Rate Plot (ax2)
  if "respiratory_waveform" in vital_signs:
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Waveform (unitless)', color=METHOD_COLORS[args.method])
    ax2.tick_params(axis='y', labelcolor=METHOD_COLORS[args.method])
    rr_string = f" -> Global RR: {vital_signs['respiratory_rate']['value']:.1f} bpm" if "respiratory_rate" in vital_signs else ""
    ax2.plot(vital_signs['respiratory_waveform']['data'], color=METHOD_COLORS[args.method], label=f"Respiratory Waveform{rr_string}", zorder=10)
    ax2.plot(vital_signs['respiratory_waveform']['confidence'], color=METHOD_COLORS[args.method], linestyle='--', label='Respiratory Confidence', zorder=5)
    if resp_gt is not None:
      rr_gt = estimate_rr_from_signal(signal=resp_gt, f_s=fps, scope=EScope.GLOBAL, method=EMethod.PERIODOGRAM)
      ax2.plot(resp_gt, color=COLOR_GT, label=f"Ground Truth Respiration -> RR: {rr_gt:.1f} bpm", zorder=0)
    ax2_handles, ax2_labels = ax2.get_legend_handles_labels()
    if "rolling_respiratory_rate" in vital_signs:
      ax2_rr = ax2.twinx()
      rr_color = '#d62728'
      ax2_rr.set_ylabel('Respiratory Rate (bpm)', color=rr_color)
      ax2_rr.tick_params(axis='y', labelcolor=rr_color)
      ax2_rr.set_ylim(0, 45)
      rolling_rr_data = vital_signs['rolling_respiratory_rate']['data']
      line_rr, = ax2_rr.plot(rolling_rr_data, color=rr_color, linestyle='-', label='Rolling Respiratory Rate')
      ax2_handles.append(line_rr)
      ax2_labels.append('Rolling Respiratory Rate')
    ax2.legend(ax2_handles, ax2_labels, loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  plt.savefig('results.png')
  plt.show()

def method_type(name):
  try:
    return Method[name]
  except KeyError:
    raise argparse.ArgumentTypeError(f"{name} is not a valid Method")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--api_key', type=str, default='', help='Your API key. Get one for free at https://www.rouast.com/api.')
  parser.add_argument('--vitals_path', type=str, default=None, help='Path to ground truth vitals')
  parser.add_argument('--video_path', type=str, default='examples/sample_video_1.mp4', help='Path to video')
  parser.add_argument('--method', type=method_type, default='VITALLENS', help='Choice of method')
  parser.add_argument('--input_str', type=str2bool, default=True, help='If true, pass filepath to VitalLens, otherwise read video into memory first')
  args = parser.parse_args()
  run(args)
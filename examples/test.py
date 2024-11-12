import sys
sys.path.append('../vitallens-python')
import argparse
import matplotlib.pyplot as plt
import os
import pandas as pd
from prpy.constants import SECONDS_PER_MINUTE
from prpy.ffmpeg.probe import probe_video
from prpy.ffmpeg.readwrite import read_video_from_path
from prpy.helpers import str2bool
from prpy.numpy.signal import estimate_freq
import timeit
from vitallens import VitalLens, Method
from vitallens.utils import download_file
from vitallens.constants import CALC_HR_MIN, CALC_HR_MAX
from vitallens.constants import CALC_RR_MIN, CALC_RR_MAX

COLOR_GT = '#000000'
METHOD_COLORS = {
  Method.VITALLENS: '#00a4df',
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
  vitals = pd.read_csv(args.vitals_path) if os.path.exists(args.vitals_path) else []
  ppg_gt = vitals['ppg'] if 'ppg' in vitals else None 
  resp_gt = vitals['resp'] if 'resp' in vitals else None
  # Get video
  fps, *_ = probe_video(args.video_path)
  if args.input_str:
    video = args.video_path
    print("Using video at: {}".format(args.video_path))
  else:
    print("Reading full video into memory from {}...".format(args.video_path))
    video, _ = read_video_from_path(path=args.video_path, pix_fmt='rgb24')
    print("Video shape: {}".format(video.shape))
  # Estimate vitals and measure inference time
  vl = VitalLens(method=args.method, api_key=args.api_key)
  start = timeit.default_timer()
  result = vl(video=video, fps=fps)
  stop = timeit.default_timer()
  time_ms = (stop-start)*1000
  print("Inference time: {:.2f} ms".format(time_ms))
  # Print the results
  print(result)
  # Plot the results
  vital_signs = result[0]['vital_signs']
  if "respiratory_waveform" in vital_signs:
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 6))
  else:
    fig, ax1 = plt.subplots(1, figsize=(12, 6))
  fig.suptitle('Vital signs estimated from {} using {} in {:.2f} ms'.format(args.video_path, args.method.name, time_ms))
  if "ppg_waveform" in vital_signs and ppg_gt is not None:
    hr_gt = estimate_freq(ppg_gt, f_s=fps, f_res=0.005, f_range=(CALC_HR_MIN/SECONDS_PER_MINUTE, CALC_HR_MAX/SECONDS_PER_MINUTE), method='periodogram') * SECONDS_PER_MINUTE
    ax1.plot(ppg_gt, color=COLOR_GT, label='PPG Waveform Ground Truth -> HR: {:.1f} bpm'.format(hr_gt))
  if "respiratory_waveform" in vital_signs and resp_gt is not None:
    rr_gt = estimate_freq(resp_gt, f_s=fps, f_res=0.005, f_range=(CALC_RR_MIN/SECONDS_PER_MINUTE, CALC_RR_MAX/SECONDS_PER_MINUTE), method='periodogram') * SECONDS_PER_MINUTE
    ax2.plot(resp_gt, color=COLOR_GT, label='Respiratory Waveform Ground Truth -> RR: {:.1f} bpm'.format(rr_gt))
  if "ppg_waveform" in vital_signs:
    ax1.plot(vital_signs['ppg_waveform']['data'], color=METHOD_COLORS[args.method], label='PPG Waveform Estimate -> HR: {:.1f} bpm ({:.0f}% confidence)'.format(
      vital_signs['heart_rate']['value'], vital_signs['heart_rate']['confidence']*100))
    ax1.plot(vital_signs['ppg_waveform']['confidence'], color=METHOD_COLORS[args.method], label='PPG Waveform Estimation Confidence')
  if "respiratory_waveform" in vital_signs:
    ax2.plot(vital_signs['respiratory_waveform']['data'], color=METHOD_COLORS[args.method], label='Respiratory Waveform Estimate -> RR: {:.1f} bpm ({:.0f}% confidence)'.format(
      vital_signs['respiratory_rate']['value'], vital_signs['respiratory_rate']['confidence']*100))
    ax2.plot(vital_signs['respiratory_waveform']['confidence'], color=METHOD_COLORS[args.method], label='Respiratory Waveform Estimation Confidence')
  ax1.legend()
  if 'respiratory_waveform' in vital_signs: ax2.legend()
  plt.show()
  plt.savefig('results.png')

def method_type(name):
  try:
    return Method[name]
  except KeyError:
    raise argparse.ArgumentTypeError(f"{name} is not a valid Method")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--api_key', type=str, default='', help='Your API key. Get one for free at https://www.rouast.com/api.')
  parser.add_argument('--vitals_path', type=str, default='examples/sample_vitals_1.csv', help='Path to ground truth vitals')
  parser.add_argument('--video_path', type=str, default='examples/sample_video_1.mp4', help='Path to video')
  parser.add_argument('--method', type=method_type, default='VITALLENS', help='Choice of method')
  parser.add_argument('--input_str', type=str2bool, default=True, help='If true, pass filepath to VitalLens, otherwise read video into memory first')
  args = parser.parse_args()
  run(args)

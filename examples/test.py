import sys
sys.path.append('../vitallens-python')
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from prpy.ffmpeg.probe import probe_video
from prpy.ffmpeg.readwrite import read_video_from_path
from prpy.helpers import str2bool
from prpy.numpy.signal import estimate_freq
import timeit
from vitallens import VitalLens, Method

COLOR_GT = '#000000'
METHOD_COLORS = {
  Method.VITALLENS: '#00a4df',
  Method.G: '#00ff00',
  Method.CHROM: '#4ceaff',
  Method.POS: '#23b031'
}

def run(args=None):
  # Get ground truth vitals
  vitals = pd.read_csv(args.vitals_path)
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
  # Plot the results
  if 'resp' in result[0]:
    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(12, 6))
  else:
    fig, ax1 = plt.subplots(1, figsize=(12, 6))
  fig.suptitle('Vital signs estimated from {} using {} in {:.2f} ms'.format(args.video_path, args.method.name, time_ms))
  if "pulse" in result[0] and ppg_gt is not None:
    hr_gt = estimate_freq(ppg_gt, f_s=fps, f_res=0.005, f_range=(40./60., 240./60.), method='periodogram') * 60.
    ax1.plot(ppg_gt, color=COLOR_GT, label='Pulse Ground Truth -> HR: {:.1f} bpm'.format(hr_gt))
  if "resp" in result[0] and resp_gt is not None:
    rr_label = estimate_freq(resp_gt, f_s=fps, f_res=0.005, f_range=(4./60., 90./60.), method='periodogram') * 60.
    ax2.plot(resp_gt, color=COLOR_GT, label='Resp Ground Truth -> RR: {:.1f} bpm'.format(rr_label))
  if "pulse" in result[0]:
    hr = estimate_freq(result[0]['pulse']['val'], f_s=fps, f_res=0.005, f_range=(40./60., 240./60.), method='periodogram') * 60.
    ax1.plot(result[0]['pulse']['val'], color=METHOD_COLORS[args.method], label='Pulse Estimate -> HR: {:.1f} bpm'.format(hr))
    ax1.plot(result[0]['pulse']['conf'], color=METHOD_COLORS[args.method], label='Pulse Estimation Confidence')
  if "resp" in result[0]:
    rr = estimate_freq(result[0]['resp']['val'], f_s=fps, f_res=0.005, f_range=(4./60., 90./60.), method='periodogram') * 60.
    ax2.plot(result[0]['resp']['val'], color=METHOD_COLORS[args.method], label='Resp Estimate -> RR: {:.1f} bpm'.format(rr))
    ax2.plot(result[0]['resp']['conf'], color=METHOD_COLORS[args.method], label='Resp Estimation Confidence')
  ax1.legend()
  if 'resp' in result[0]: ax2.legend()
  plt.show()

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

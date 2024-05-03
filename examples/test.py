import sys
sys.path.append('../vitallens-python')
import vitallens
import timeit

from prpy.ffmpeg.probe import probe_video
from prpy.ffmpeg.readwrite import read_video_from_path

input_str = True
video_path = "examples/test.mp4"

if input_str:
  fps = None
  video = video_path
  print("Using video at: {}".format(video_path))
else:
  fps, *_ = probe_video(video_path)
  print("Reading full video from {}...".format(video_path))
  video, _ = read_video_from_path(path=video_path, pix_fmt='rgb24')
  print("Video shape: {}".format(video.shape))

vl = vitallens.VitalLens(
  method=vitallens.Method.POS,
  api_key="INSERT_API_KEY_HERE")
start = timeit.default_timer()
result = vl(video=video, fps=fps, override_fps_target=30.0)
stop = timeit.default_timer()

print("Inference time: {:.2f} ms".format((stop-start)*1000))

import matplotlib.pyplot as plt
if 'pulse' in result[0]: plt.plot(result[0]['pulse']['sig'])
if 'resp' in result[0]: plt.plot(result[0]['resp']['sig'])
plt.show()

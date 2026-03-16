# Examples & Usage Recipes

<!-- mkdocs-start -->
The `examples/` folder contains sample scripts and video data to help you evaluate `vitallens` against ground truth data, run it in Docker, or integrate it into your own pipeline.

## Real-time Webcam Demo (`live.py`)

This script opens your webcam and streams frames to the API in chunks.

```bash
pip install opencv-python
python examples/live.py --method=vitallens --api_key=YOUR_API_KEY
```

## Evaluation Script (`test.py`)

This directory contains sample scripts and video data to help you evaluate `vitallens` against ground truth data, run it in Docker, or integrate it into your own pipeline.

### Included Sample Data
We have included two videos with synchronized ground truth data:

* **`sample_video_1.mp4`**: Includes ground truth for PPG, ECG, Respiratory Waveform, Blood Pressure, SpO2, Heart Rate, and HRV.
* **`sample_video_2.mp4`**: Sourced from the [VitalVideos](http://vitalvideos.org) dataset. Includes ground truth for PPG.

### Usage

First, install the visualization dependencies:
```bash
pip install matplotlib pandas
```

Run the comparison using the VitalLens API:

```bash
python examples/test.py \
  --method=vitallens \
  --api_key="YOUR_API_KEY" \
  --video_path=examples/sample_video_2.mp4 \
  --vitals_path=examples/sample_vitals_2.csv
```

You can also benchmark local methods (no API key required):

```bash
python examples/test.py --method=POS --video_path=examples/sample_video_1.mp4
```

## Integration Recipes

Copy-paste patterns for common integration scenarios.

### Processing a Video File

The simplest way to use the library.

```python
from vitallens import VitalLens, Method

# Initialize
vl = VitalLens(method="vitallens", api_key="YOUR_API_KEY")

# Run inference
results = vl("path/to/video.mp4")

# Access results
print("Heart Rate:", results[0]['vitals']['heart_rate']['value'])
```

### Processing Raw Frames (Numpy/OpenCV)

Use this if you are already reading video frames with OpenCV or `imageio`.

```python
import cv2
from vitallens import VitalLens, Method

vl = VitalLens(method="vitallens", api_key="YOUR_API_KEY")

# 1. Read video into a list of frames
frames = []
cap = cv2.VideoCapture("video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    # Convert BGR (OpenCV default) to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)
cap.release()

# 2. Convert to numpy array (N, H, W, 3)
import numpy as np
video_arr = np.array(frames)

# 3. Run inference (must provide fps explicitly)
results = vl(video_arr, fps=fps)
```

## Running with Docker

If you encounter dependency issues (e.g., with `onnxruntime` or `ffmpeg`), you can run the example scripts inside our Docker container.

**1. Build the image**

```bash
docker build -t vitallens .
```

**2. Run the evaluation**

```bash
docker run vitallens \
  --api_key "YOUR_API_KEY" \
  --vitals_path "examples/sample_vitals_2.csv" \
  --video_path "examples/sample_video_2.mp4" \
  --method "vitallens"
```

**3. Extract the plot**
Since the plot cannot display inside the container, copy it out after running:

```bash
docker cp <container_id>:/app/results.png .
```

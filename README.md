<div align="center">
  <a href="https://www.rouast.com/api/">
    <img src="./assets/logo.svg" alt="VitalLens API Logo" height="80px" width="80px"/>
  </a>
  <h1>vitallens-python</h1>
  <p align="center">
    <p>Estimate vital signs such as heart rate and respiratory rate from video in Python.</p>
  </p>

[![Tests](https://github.com/Rouast-Labs/vitallens-python/actions/workflows/main.yml/badge.svg)](https://github.com/Rouast-Labs/vitallens-python/actions/workflows/main.yml)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/vitallens?period=total&units=international_system&left_color=grey&right_color=blue&left_text=pip%20downloads)](https://pypi.org/project/vitallens/)
[![Website](https://img.shields.io/badge/Website-rouast.com/api-blue.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+Cjxzdmcgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgdmlld0JveD0iMCAwIDI0IDI0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHhtbDpzcGFjZT0icHJlc2VydmUiIHhtbG5zOnNlcmlmPSJodHRwOi8vd3d3LnNlcmlmLmNvbS8iIHN0eWxlPSJmaWxsLXJ1bGU6ZXZlbm9kZDtjbGlwLXJ1bGU6ZXZlbm9kZDtzdHJva2UtbGluZWpvaW46cm91bmQ7c3Ryb2tlLW1pdGVybGltaXQ6MjsiPgogICAgPGcgdHJhbnNmb3JtPSJtYXRyaXgoMC4xODc5OTgsMCwwLDAuMTg3OTk4LDIzLjMyOTYsMTIuMjQ1MykiPgogICAgICAgIDxwYXRoIGQ9Ik0wLC0yLjgyOEMwLjMzOSwtMi41OTYgMC42NzQsLTIuMzk3IDEuMDA1LC0yLjIyNkwzLjU2NiwtMTUuODczQzAuMjY5LC0yMy42NTYgLTMuMTc1LC0zMS42MTUgLTkuNjU1LC0zMS42MTVDLTE2LjQ2MiwtMzEuNjE1IC0xNy41NDgsLTIzLjk0MiAtMTkuOTQ3LDAuMzEyQy0yMC40MjEsNS4wODEgLTIxLjAzOCwxMS4zMDggLTIxLjcxMSwxNi4wMzFDLTI0LjAxNiwxMS45NTQgLTI2LjY3NSw2LjU0OSAtMjguNDIsMy4wMDJDLTMzLjQ3OSwtNy4yNzggLTM0LjY2NSwtOS4zOTQgLTM2Ljg4OCwtMTAuNTM0Qy0zOS4wMzMsLTExLjYzOSAtNDAuOTk1LC0xMS41OTEgLTQyLjM3MSwtMTEuNDA4Qy00My4wMzcsLTEzIC00My45NDQsLTE1LjQzMSAtNDQuNjY4LC0xNy4zNjdDLTQ5LjUyOSwtMzAuMzkxIC01MS43NzIsLTM1LjQxMiAtNTYuMDY2LC0zNi40NTNDLTU3LjU2NiwtMzYuODE3IC01OS4xNDYsLTM2LjQ5MSAtNjAuMzk5LC0zNS41NjJDLTYzLjQyOCwtMzMuMzI0IC02NC4wMTYsLTI5LjYwMSAtNjUuNjUsLTIuMzcxQy02Ni4wMTcsMy43NDcgLTY2LjQ5NSwxMS43MTMgLTY3LjA1NiwxNy43NzZDLTY5LjE4MiwxNC4xMDggLTcxLjUyNiw5Ljc4MiAtNzMuMjY5LDYuNTcxQy04MS4wNTgsLTcuNzk0IC04Mi42ODcsLTEwLjQyMiAtODUuNzE5LC0xMS4zMUMtODcuNjQ2LC0xMS44NzcgLTg5LjIyMywtMTEuNjYgLTkwLjQyNSwtMTEuMjQ0Qy05MS4yOTYsLTEzLjM3NCAtOTIuNDM0LC0xNi45NzkgLTkzLjI1NSwtMTkuNTgzQy05Ni42LC0zMC4xODkgLTk4LjYyLC0zNi41ODggLTEwNC4xMzUsLTM2LjU4OEMtMTEwLjQ4NCwtMzYuNTg4IC0xMTAuODQzLC0zMC4zOTEgLTExMi4zNTUsLTQuMzExQy0xMTIuNzA3LDEuNzUgLTExMy4xNjksOS43NDIgLTExMy43NDEsMTUuNTUxQy0xMTYuMywxMS43ODEgLTExOS4yOSw2Ljk3OSAtMTIxLjQ1LDMuNDlMLTEyNC4wOTUsMTcuNTc2Qy0xMTcuNjA3LDI3LjU4NSAtMTE0Ljc2NiwzMC40NTggLTExMS4yMDQsMzAuNDU4Qy0xMDQuNjAzLDMwLjQ1OCAtMTA0LjIyMiwyMy44OTMgLTEwMi42MjEsLTMuNzQ3Qy0xMDIuNDIyLC03LjE3IC0xMDIuMTk3LC0xMS4wNDYgLTEwMS45NDYsLTE0LjcyOUMtOTkuNTUxLC03LjIxNiAtOTguMTkyLC0zLjY4NSAtOTUuNTQxLC0yLjA1Qy05Mi42OTgsLTAuMjk3IC05MC4zOTgsLTAuNTQ3IC04OC44MTMsLTEuMTU3Qy04Ny4wNCwxLjYyOSAtODQuMTExLDcuMDMgLTgxLjg0LDExLjIyQy03MS45NTUsMjkuNDQ2IC02OS4yMDIsMzMuNzM1IC02NC44NDYsMzMuOTc1Qy02NC42NjEsMzMuOTg1IC02NC40OCwzMy45ODkgLTY0LjMwNSwzMy45ODlDLTU4LjA2NCwzMy45ODkgLTU3LjY2MiwyNy4zMDQgLTU1LjkxNywtMS43ODdDLTU1LjYzMSwtNi41MyAtNTUuMywtMTIuMDcgLTU0LjkyNywtMTYuOTQ4Qy01NC41MTIsLTE1Ljg1MiAtNTQuMTI5LC0xNC44MjkgLTUzLjgwMywtMTMuOTU1Qy01MS4wNTYsLTYuNTk0IC01MC4xODcsLTQuNDExIC00OC40NzMsLTMuMDQyQy00NS44NywtMC45NjIgLTQzLjE0OSwtMS4zNjkgLTQxLjczNywtMS42MjhDLTQwLjYwMiwwLjMyOSAtMzguNjY0LDQuMjcxIC0zNy4xNjksNy4zMDZDLTI4LjgyNSwyNC4yNjQgLTI1LjE2OCwzMC42NzMgLTE5LjgxMiwzMC42NzNDLTEzLjE1NSwzMC42NzMgLTEyLjM2MiwyMi42NjYgLTEwLjI0NCwxLjI3MkMtOS42NjMsLTQuNjA2IC04Ljg4MiwtMTIuNDk2IC03Ljk5NiwtMTcuODMxQy02Ljk2MywtMTUuNzI5IC01Ljk1NCwtMTMuMzUgLTUuMzA3LC0xMS44MkMtMy4xNDUsLTYuNzIxIC0yLjAxNywtNC4yMDkgMCwtMi44MjgiIHN0eWxlPSJmaWxsOnJnYigwLDE2NCwyMjQpO2ZpbGwtcnVsZTpub256ZXJvOyIvPgogICAgPC9nPgo8L3N2Zz4K)](https://www.rouast.com/api/)
[![Documentation](https://img.shields.io/badge/Docs-docs.rouast.com-blue.svg?logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8+CjwhRE9DVFlQRSBzdmcgUFVCTElDICItLy9XM0MvL0RURCBTVkcgMS4xLy9FTiIgImh0dHA6Ly93d3cudzMub3JnL0dyYXBoaWNzL1NWRy8xLjEvRFREL3N2ZzExLmR0ZCI+Cjxzdmcgd2lkdGg9IjEwMCUiIGhlaWdodD0iMTAwJSIgdmlld0JveD0iMCAwIDI0IDI0IiB2ZXJzaW9uPSIxLjEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHhtbDpzcGFjZT0icHJlc2VydmUiIHhtbG5zOnNlcmlmPSJodHRwOi8vd3d3LnNlcmlmLmNvbS8iIHN0eWxlPSJmaWxsLXJ1bGU6ZXZlbm9kZDtjbGlwLXJ1bGU6ZXZlbm9kZDtzdHJva2UtbGluZWpvaW46cm91bmQ7c3Ryb2tlLW1pdGVybGltaXQ6MjsiPgogICAgPGcgdHJhbnNmb3JtPSJtYXRyaXgoMC4xODc5OTgsMCwwLDAuMTg3OTk4LDIzLjMyOTYsMTIuMjQ1MykiPgogICAgICAgIDxwYXRoIGQ9Ik0wLC0yLjgyOEMwLjMzOSwtMi41OTYgMC42NzQsLTIuMzk3IDEuMDA1LC0yLjIyNkwzLjU2NiwtMTUuODczQzAuMjY5LC0yMy42NTYgLTMuMTc1LC0zMS42MTUgLTkuNjU1LC0zMS42MTVDLTE2LjQ2MiwtMzEuNjE1IC0xNy41NDgsLTIzLjk0MiAtMTkuOTQ3LDAuMzEyQy0yMC40MjEsNS4wODEgLTIxLjAzOCwxMS4zMDggLTIxLjcxMSwxNi4wMzFDLTI0LjAxNiwxMS45NTQgLTI2LjY3NSw2LjU0OSAtMjguNDIsMy4wMDJDLTMzLjQ3OSwtNy4yNzggLTM0LjY2NSwtOS4zOTQgLTM2Ljg4OCwtMTAuNTM0Qy0zOS4wMzMsLTExLjYzOSAtNDAuOTk1LC0xMS41OTEgLTQyLjM3MSwtMTEuNDA4Qy00My4wMzcsLTEzIC00My45NDQsLTE1LjQzMSAtNDQuNjY4LC0xNy4zNjdDLTQ5LjUyOSwtMzAuMzkxIC01MS43NzIsLTM1LjQxMiAtNTYuMDY2LC0zNi40NTNDLTU3LjU2NiwtMzYuODE3IC01OS4xNDYsLTM2LjQ5MSAtNjAuMzk5LC0zNS41NjJDLTYzLjQyOCwtMzMuMzI0IC02NC4wMTYsLTI5LjYwMSAtNjUuNjUsLTIuMzcxQy02Ni4wMTcsMy43NDcgLTY2LjQ5NSwxMS43MTMgLTY3LjA1NiwxNy43NzZDLTY5LjE4MiwxNC4xMDggLTcxLjUyNiw5Ljc4MiAtNzMuMjY5LDYuNTcxQy04MS4wNTgsLTcuNzk0IC04Mi42ODcsLTEwLjQyMiAtODUuNzE5LC0xMS4zMUMtODcuNjQ2LC0xMS44NzcgLTg5LjIyMywtMTEuNjYgLTkwLjQyNSwtMTEuMjQ0Qy05MS4yOTYsLTEzLjM3NCAtOTIuNDM0LC0xNi45NzkgLTkzLjI1NSwtMTkuNTgzQy05Ni42LC0zMC4xODkgLTk4LjYyLC0zNi41ODggLTEwNC4xMzUsLTM2LjU4OEMtMTEwLjQ4NCwtMzYuNTg4IC0xMTAuODQzLC0zMC4zOTEgLTExMi4zNTUsLTQuMzExQy0xMTIuNzA3LDEuNzUgLTExMy4xNjksOS43NDIgLTExMy43NDEsMTUuNTUxQy0xMTYuMywxMS43ODEgLTExOS4yOSw2Ljk3OSAtMTIxLjQ1LDMuNDlMLTEyNC4wOTUsMTcuNTc2Qy0xMTcuNjA3LDI3LjU4NSAtMTE0Ljc2NiwzMC40NTggLTExMS4yMDQsMzAuNDU4Qy0xMDQuNjAzLDMwLjQ1OCAtMTA0LjIyMiwyMy44OTMgLTEwMi42MjEsLTMuNzQ3Qy0xMDIuNDIyLC03LjE3IC0xMDIuMTk3LC0xMS4wNDYgLTEwMS45NDYsLTE0LjcyOUMtOTkuNTUxLC03LjIxNiAtOTguMTkyLC0zLjY4NSAtOTUuNTQxLC0yLjA1Qy05Mi42OTgsLTAuMjk3IC05MC4zOTgsLTAuNTQ3IC04OC44MTMsLTEuMTU3Qy04Ny4wNCwxLjYyOSAtODQuMTExLDcuMDMgLTgxLjg0LDExLjIyQy03MS45NTUsMjkuNDQ2IC02OS4yMDIsMzMuNzM1IC02NC44NDYsMzMuOTc1Qy02NC42NjEsMzMuOTg1IC02NC40OCwzMy45ODkgLTY0LjMwNSwzMy45ODlDLTU4LjA2NCwzMy45ODkgLTU3LjY2MiwyNy4zMDQgLTU1LjkxNywtMS43ODdDLTU1LjYzMSwtNi41MyAtNTUuMywtMTIuMDcgLTU0LjkyNywtMTYuOTQ4Qy01NC41MTIsLTE1Ljg1MiAtNTQuMTI5LC0xNC44MjkgLTUzLjgwMywtMTMuOTU1Qy01MS4wNTYsLTYuNTk0IC01MC4xODcsLTQuNDExIC00OC40NzMsLTMuMDQyQy00NS44NywtMC45NjIgLTQzLjE0OSwtMS4zNjkgLTQxLjczNywtMS42MjhDLTQwLjYwMiwwLjMyOSAtMzguNjY0LDQuMjcxIC0zNy4xNjksNy4zMDZDLTI4LjgyNSwyNC4yNjQgLTI1LjE2OCwzMC42NzMgLTE5LjgxMiwzMC42NzNDLTEzLjE1NSwzMC42NzMgLTEyLjM2MiwyMi42NjYgLTEwLjI0NCwxLjI3MkMtOS42NjMsLTQuNjA2IC04Ljg4MiwtMTIuNDk2IC03Ljk5NiwtMTcuODMxQy02Ljk2MywtMTUuNzI5IC01Ljk1NCwtMTMuMzUgLTUuMzA3LC0xMS44MkMtMy4xNDUsLTYuNzIxIC0yLjAxNywtNC4yMDkgMCwtMi44MjgiIHN0eWxlPSJmaWxsOnJnYigwLDE2NCwyMjQpO2ZpbGwtcnVsZTpub256ZXJvOyIvPgogICAgPC9nPgo8L3N2Zz4K)](https://docs.rouast.com/)

</div>

`vitallens` is a Python client for the [**VitalLens API**](https://www.rouast.com/api/), using the same inference engine as our [free iOS app VitalLens](https://apps.apple.com/us/app/vitallens/id6472757649).
Furthermore, it includes fast implementations of several other heart rate estimation methods from video such as `G`, `CHROM`, and `POS`.

- Accepts as input either a video filepath or an in-memory video as `np.ndarray`
- Performs fast face detection if required - you can also pass existing detections
- `vitallens.Method.VITALLENS` supports *heart rate*, *respiratory rate*, *pulse waveform*, and *respiratory waveform* estimation. In addition, it returns an estimation confidence for each vital. We are working to support more vital signs in the future.
- `vitallens.Method.{G/CHROM/POS}` support faster, but less accurate *heart rate* and *pulse waveform* estimation.
- While `VITALLENS` requires an API Key, `G`, `CHROM`, and `POS` do not. [Register on our website to get a free API Key.](https://www.rouast.com/api/)

Estimate vitals in a few lines of code:

```python
from vitallens import VitalLens, Method

vl = VitalLens(method=Method.VITALLENS, api_key="YOUR_API_KEY")
result = vl("video.mp4")
print(result)
```

Using a different language or platform? We also have a [JavaScript client](https://github.com/Rouast-Labs/vitallens.js).

### Disclaimer

`vitallens` provides vital sign estimates for general wellness purposes only. It is not intended for medical use. Always consult with your doctor for any health concerns or for medically precise measurement.

See also our [Terms of Service for the VitalLens API](https://www.rouast.com/api/terms) and our [Privacy Policy](https://www.rouast.com/privacy).

## Installation

General prerequisites are `python>=3.9` and `ffmpeg` installed and accessible via the `$PATH` environment variable.

The easiest way to install the latest version of `vitallens` and its Python dependencies:

```
pip install vitallens
```

Alternatively, it can be done by cloning the source:

```
git clone https://github.com/Rouast-Labs/vitallens-python.git
pip install ./vitallens-python
```

### Dealing with possible issues

On newer versions of Python you may face the issue that the dependency `onnxruntime` cannot be installed via pip. If you are using `conda`, you can try installing it via `conda install -c conda-forge onnxruntime`, and then run `pip install vitallens` again. Otherwise try using Python 3.9, 3.10, or 3.11.

## How to use

To start using `vitallens`, first create an instance of `vitallens.VitalLens`. 
It can be configured using the following parameters:

| Parameter               | Description                                                                        | Default            |
|-------------------------|------------------------------------------------------------------------------------|--------------------|
| method                  | Inference method. {`Method.VITALLENS`, `Method.POS`, `Method.CHROM` or `Method.G`} | `Method.VITALLENS` |
| mode                    | Operation mode. {`Mode.BATCH` for indep. videos or `Mode.BURST` for video stream}  | `Mode.BATCH`       |
| api_key                 | Usage key for the VitalLens API (required for `Method.VITALLENS`)                  | `None`             |
| detect_faces            | `True` if faces need to be detected, otherwise `False`.                            | `True`             |
| estimate_rolling_vitals | Set `True` to compute rolling vitals (e.g., `rolling_heart_rate`).                 | `True`             |
| fdet_max_faces          | The maximum number of faces to detect (if necessary).                              | `1`                |
| fdet_fs                 | Frequency [Hz] at which faces should be scanned - otherwise linearly interpolated. | `1.0`              |
| export_to_json          | If `True`, write results to a json file.                                           | `True`             |
| export_dir              | The directory to which json files are written.                                     | `.`                |

Once instantiated, `vitallens.VitalLens` can be called to estimate vitals.
In `Mode.BATCH` calls are assumed to be working on independent videos, whereas in `Mode.BURST` we expect the subsequent calls to pass the next frames of the same video (stream) as `np.ndarray`.
Calls are configured using the following parameters:

| Parameter           | Description                                                                           | Default |
|---------------------|---------------------------------------------------------------------------------------|---------|
| video               | The video to analyze. Either a path to a video file or `np.ndarray`. [More info here.](https://github.com/Rouast-Labs/vitallens-python/raw/main/vitallens/client.py#L114)    |         |
| faces               | Face detections. Ignored unless `detect_faces=False`. [More info here.](https://github.com/Rouast-Labs/vitallens-python/raw/main/vitallens/client.py#L117) | `None`  |
| fps                 | Sampling frequency of the input video. Required if video is `np.ndarray`.             | `None`  |
| override_fps_target | Target frequency for inference (optional - use methods's default otherwise).          | `None`  |
| export_filename     | Filename for json export if applicable.                                               | `None`  |

`vitallens` returns estimates of the following vital signs if using `Mode.BATCH` with a minimum of 16 frames:

| Name                       | Type                | Returned if                                                                                              |
|----------------------------|---------------------|----------------------------------------------------------------------------------------------------------|
| `heart_rate`               | Global value        | Video at least 2 seconds long and using `Method.VITALLENS`, `Method.POS`, `Method.CHROM` or `Method.G`   |
| `rolling_heart_rate`       | Continuous values   | Video more than 10 seconds long and using `Method.VITALLENS`, `Method.POS`, `Method.CHROM` or `Method.G` and `estimate_rolling_vitals=True` |
| `ppg_waveform`             | Continuous waveform | Using `Method.VITALLENS`, `Method.POS`, `Method.CHROM` or `Method.G`                                     |
| `respiratory_rate`         | Global value        | Video at least 4 seconds long and using `Method.VITALLENS`                                               |
| `rolling_respiratory_rate` | Continuous values   | Video more than 30 seconds long and using `Method.VITALLENS` and `estimate_rolling_vitals=True`          |
| `respiratory_waveform`     | Continuous waveform | Using `Method.VITALLENS`                                                                                 |

The estimation results are returned as a `list`. It contains a `dict` for each distinct face, with the following structure:

```
[
  {
    'face': {
      'coordinates': <Face coordinates for each frame as np.ndarray of shape (n_frames, 4)>,
      'confidence': <Face live confidence for each frame as np.ndarray of shape (n_frames,)>,
      'note': <Explanatory note>
    },
    'vital_signs': {
      'heart_rate': {
        'value': <Estimated global value as float scalar>,
        'unit': <Value unit>,
        'confidence': <Estimation confidence as float scalar>,
        'note': <Explanatory note>
      },
      'respiratory_rate': {
        'value': <Estimated global value as float scalar>,
        'unit': <Value unit>,
        'confidence': <Estimation confidence as float scalar>,
        'note': <Explanatory note>
      },
      'ppg_waveform': {
        'data': <Estimated waveform value for each frame as np.ndarray of shape (n_frames,)>,
        'unit': <Data unit>,
        'confidence': <Estimation confidence for each frame as np.ndarray of shape (n_frames,)>,
        'note': <Explanatory note>
      },
      'respiratory_waveform': {
        'data': <Estimated waveform value for each frame as np.ndarray of shape (n_frames,)>,
        'unit': <Data unit>,
        'confidence': <Estimation confidence for each frame as np.ndarray of shape (n_frames,)>,
        'note': <Explanatory note>
      },
      'rolling_heart_rate': {
        'data': <Estimated value for each frame as np.ndarray of shape (n_frames,)>,
        'unit': <Value unit>,
        'confidence': <Estimation confidence for each frame as np.ndarray of shape (n_frames,)>,
        'note': <Explanatory note>
      },
      'rolling_respiratory_rate': {
        'data': <Estimated value for each frame as np.ndarray of shape (n_frames,)>,
        'unit': <Value unit>,
        'confidence': <Estimation confidence for each frame as np.ndarray of shape (n_frames,)>,
        'note': <Explanatory note>
      }
    },
    "message": <Message about estimates>
  },
  { 
    <same structure for face 2 if present>
  },
  ...
]
```

## Examples to get started

### Live test with webcam in real-time

Test `vitallens` in real-time with your webcam using the script `examples/live.py`.
This uses `Mode.BURST` to update results continuously (approx. every 2 seconds for `Method.VITALLENS`).
Some options are available:

- `method`: Choose from [`VITALLENS`, `POS`, `G`, `CHROM`] (Default: `VITALLENS`)
- `api_key`: Pass your API Key. Required if using `method=VITALLENS`.

May need to install requirements first: `pip install opencv-python`

```
python examples/live.py --method=VITALLENS --api_key=YOUR_API_KEY
```

### Compare results with gold-standard labels using our example script

There is an example Python script in `examples/test.py` which uses `Mode.BATCH` to run vitals estimation and plot the predictions against ground truth labels recorded with gold-standard medical equipment.
Some options are available:

- `method`: Choose from [`VITALLENS`, `POS`, `G`, `CHROM`] (Default: `VITALLENS`)
- `video_path`: Path to video (Default: `examples/sample_video_1.mp4`)
- `vitals_path`: Path to gold-standard vitals (Default: `examples/sample_vitals_1.csv`)
- `api_key`: Pass your API Key. Required if using `method=VITALLENS`.

May need to install requirements first: `pip install matplotlib pandas`

For example, to reproduce the results from the banner image on the [VitalLens API Webpage](https://www.rouast.com/api/):

```
python examples/test.py --method=VITALLENS --video_path=examples/sample_video_2.mp4 --vitals_path=examples/sample_vitals_2.csv --api_key=YOUR_API_KEY
```

This sample is kindly provided by the [VitalVideos](http://vitalvideos.org) dataset.

### Use VitalLens API to estimate vitals from a video file

```python
from vitallens import VitalLens, Method

vl = VitalLens(method=Method.VITALLENS, api_key="YOUR_API_KEY")
result = vl("video.mp4")
```

### Use POS method on an `np.ndarray` of video frames

```python
from vitallens import VitalLens, Method

my_video_arr = ...
my_video_fps = 30
vl = VitalLens(method=Method.POS)
result = vl(my_video_arr, fps=my_video_fps)
```

### Run example script with Docker

If you encounter issues installing `vitallens` dependencies directly, you can use our Docker image, which contains all necessary tools and libraries.
This docker image is set up to execute the example Python script in `examples/test.py` for you. 

#### Prerequisites

- [Docker](https://docs.docker.com/engine/install/) installed on your system.

#### Usage

1. Clone the repository

```
git clone https://github.com/Rouast-Labs/vitallens-python.git && cd vitallens-python
```

2. Build the Docker image

```
docker build -t vitallens .
```

3. Run the Docker container

To run the example script on the sample video:

```
docker run vitallens \          
  --api_key "your_api_key_here" \
  --vitals_path "examples/sample_vitals_2.csv" \
  --video_path "examples/sample_video_2.mp4" \
  --method "VITALLENS"
```

You can also run it on your own video:

```
docker run vitallens \          
  --api_key "your_api_key_here" \
  --video_path "path/to/your/video.mp4" \
  --method "VITALLENS"
```

4. View the results

The results will print to the console in text form.

Please note that the example script plots won't work when running them through Docker. To to get the plot as an image file, run:

```
docker cp <container_id>:/app/results.png .
```

## Linting and tests

Before running tests, please make sure that you have an environment variable `VITALLENS_DEV_API_KEY` set to a valid API Key. 
To lint and run tests:

```
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pytest
```

## Build

To build:

```
python -m build
```

# Understanding the Results

The structure of a result, the vital keys, when each value is available and how to use the confidence scores are the same for all VitalLens clients and described in [Understanding Results](https://docs.rouast.com/guides/results/). This page covers what is specific to the Python client.

## One Result per Face

Processing a video returns a Python **list** with one result per detected face. If two people are visible, the list has two entries.

```python
results = vl(video)
# Type: List[dict]
hr = results[0]['vitals']['heart_rate']['value']
```

Waveform data, confidences and timestamps are returned as lists; scalar values are Python floats.

Besides `face`, `vitals`, `waveforms` and `message`, each result contains:

| Key | Contents |
| --- | --- |
| `rolling_vitals` | Vitals over time, see below. |
| `fps` | The frame rate the video was processed at. |
| `n` | The number of frames. |
| `time` | The timestamp of each frame in seconds. |

## Rolling Vitals

When processing a video file, the client also estimates each vital over time, as one value per frame from a trailing window (the window is listed under [Availability](https://docs.rouast.com/guides/results/#availability)). They are under `rolling_vitals`, with the same keys as `vitals`:

```python
rolling_hr = results[0]['rolling_vitals']['heart_rate']['data']  # one value per frame
```

Rolling vitals are on by default. To skip them:

```python
vl = VitalLens(estimate_rolling_vitals=False)
```

## File Export

The client also saves the results to a timestamped JSON file (e.g., `vitallens_20260203_120642.json`) in your current directory. To disable this:

```python
vl = VitalLens(export_to_json=False)
```

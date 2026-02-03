# Understanding the Results

When you process a video using the `vitallens` client, the results are returned as a standard Python **list**.

```python
results = vl(video)
# Type: List[dict]
```

## Result Structure

The list contains one dictionary **per detected face**. If two people are visible in the video, the list will have two entries.

### JSON Schema

Each face entry follows this structure. Note that strict types (like `np.ndarray`) are used for raw data, while scalar values are Python floats.

```
[
  {
    "face": {
      "coordinates": [[247, 52, 444, 332], ...],
      "confidence": [0.6115, 0.9207, 0.9183, ...],
      "note": "Face detection coordinates..."
    },
    "vital_signs": {
      "heart_rate": {
        "value": 60.5,
        "unit": "bpm",
        "confidence": 0.9242,
        "note": "Global estimate of Heart Rate..."
      },
      "respiratory_rate": {
        "value": 12.0,
        "unit": "bpm",
        "confidence": 0.9976,
        "note": "Global estimate of Respiratory Rate..."
      },
      "hrv_sdnn": {
        "value": 53.79,
        "unit": "ms",
        "confidence": 0.8712,
        "note": "Global estimate of Heart Rate Variability (SDNN)..."
      }
    },
    "message": "The provided values are estimates..."
  }
]
```

## Output & File Export

The client returns results as a Python list and, by default, also saves them to a timestamped JSON file (e.g., `vitallens_20260203_120642.json`) in your current directory.

To disable file export:

```python
vl = VitalLens(export_to_json=False)
```

## Data Availability

You might notice that not all keys (like `hrv_sdnn`) are present in every result. The availability of specific vital signs depends on **two factors**:

1. **Duration:** Physiological signals require a minimum window of time to measure accurately.
2. **Method:** Simple methods (like `pos`) only measure HR, while `vitallens` measures respiration and HRV.

| Vital Sign | Key | Type | Required Duration | Supported Methods |
| --- | --- | --- | --- | --- |
| **PPG Waveform** | `ppg_waveform` | Continuous waveform | N/A (Always) | All |
| **Heart Rate** | `heart_rate` | Global value | ≥ 5s | All |
| **Rolling HR** | `rolling_heart_rate` | Continuous values | ≥ 10s | All |
| **Respiratory Waveform** | `respiratory_waveform` | Continuous waveform | N/A (Always) | `vitallens` |
| **Respiratory Rate** | `respiratory_rate` | Global value | ≥ 10s | `vitallens` |
| **Rolling RR** | `rolling_respiratory_rate` | Continuous values | ≥ 30s | `vitallens` |
| **HRV (SDNN)** | `hrv_sdnn` | Global value | ≥ 20s | `vitallens` |
| **HRV (RMSSD)** | `hrv_rmssd` | Global value | ≥ 20s | `vitallens` |
| **HRV (LF/HF)** | `hrv_lfhf` | Global | ≥ 55s | `vitallens` |

> **Note:** Rolling metrics (continuous estimates over time) are only computed if you initialize the client with `estimate_rolling_vitals=True`.

> **Note:** HRV vitals are only available on `vitallens` version 2.0 or greater.


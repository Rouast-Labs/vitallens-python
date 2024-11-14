# Example ways to test `vitallens`

## Live test with webcam in real-time

Test `vitallens` in real-time with your webcam using the script `live.py`.
This uses `Mode.BURST` to update results continuously (approx. every 2 seconds for `Method.VITALLENS`).
Some options are available:

- `method`: Choose from [`VITALLENS`, `POS`, `G`, `CHROM`] (Default: `VITALLENS`)
- `api_key`: Pass your API Key. Required if using `method=VITALLENS`.

May need to install requirements first: `pip install opencv-python`

```
python examples/live.py --method=VITALLENS --api_key=YOUR_API_KEY
```

## Sample videos with ground truth labels

In this folder, you can find two sample videos to test `vitallens` on.
Each video has ground truth labels recorded with gold-standard medical equipment.

- `sample_video_1.mp4` which has ground truth labels for PPG Waveform (`ppg`), ECG Waveform (`ecg`), Respiratory Waveform (`resp`), Blood Pressure (`sbp` and `dbp`), Blood Oxygen (`spo2`), Heart Rate (`hr_ecg` - derived from ECG and `hr_ppg` - derived from PPG), Heart Rate Variability (`hrv_sdnn_ecg`), and Respiratory Rate (`rr`).
- `sample_video_2.mp4` which has ground truth labels for PPG Waveform (`ppg`). This sample is kindly provided by the [VitalVideos](http://vitalvideos.org) dataset.

There is a test script in `test.py` which uses `Mode.BATCH` to run vitals estimation and plot the predictions against the ground truth labels. This uses `vitallens.Mode.BATCH` mode.
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

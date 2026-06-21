# Keypoint fall-classifier training

The split is random, stratified, deterministic, and performed at video level:
75% training videos and 25% held-out test videos. MoveNet and Coral PoseNet use
the same 17 joints but require different models because their output confidence
and coordinate distributions differ. Caches live under `train/cache/windows/`
and `train/cache/pi/`.

```powershell
uv pip install --python .venv\Scripts\python.exe -r train\requirements.txt
.venv\Scripts\python.exe -m train.extract_keypoints --platform windows
.venv\Scripts\python.exe -m train.train_classifier --platform windows
```

Run Coral extraction on the Pi:

```bash
python -m train.extract_keypoints --platform pi
```

Copy `train/cache/pi/` back to Windows and run:

```powershell
.venv\Scripts\python.exe -m train.train_classifier --platform pi
```

This creates platform-specific reports and portable artifacts. Scikit-learn is
training-only and is not required by either runtime. Event tuning searches both
the probability threshold and a vote rule, such as 4 positive frames among the
last 8, so one weak Coral frame does not erase all evidence.

The Windows model is a 160-tree random forest (depth 9), with a 0.70 event
threshold and four votes among four frames. On the held-out 40 videos it
scored 92.5% accuracy and 0.930 F1, versus 82.5% and 0.821 for the old rules.
The three classifier errors were false alarms; it missed none of the 20 falls.

The Pi model is a 200-tree Extra Trees forest (depth 8), trained only from
Coral PoseNet keypoints. Its tuned event rule is 3 positive frames among the
last 4 at a 0.70 threshold. On the same held-out split it detected all 20 falls
with 7 false alarms: 82.5% accuracy, 100% recall, and 0.851 F1. Model selection
uses four-fold video-level out-of-fold predictions and F2 to favor fall recall.

# Keypoint fall-classifier training

The split is random, stratified, deterministic, and performed at video level:
75% training videos and 25% held-out test videos. Windows MoveNet extracts the
same 17 COCO keypoints used by Coral PoseNet. Cached keypoints are stored under
`train/cache/` so classifier experiments do not repeat pose inference.

```powershell
uv pip install --python .venv\Scripts\python.exe -r train\requirements.txt
.venv\Scripts\python.exe -m train.extract_keypoints
.venv\Scripts\python.exe -m train.train_classifier
```

`train/report.json` compares the winning classifier with the recorded benchmark
from the retired rules on the same held-out videos. Their implementation has
been removed. `models/fall_classifier.json` is a portable inference artifact
and does not require scikit-learn on Raspberry Pi.

The selected model is a 160-tree random forest (depth 9), with a 0.70 event
threshold and four consecutive positive frames. On the held-out 40 videos it
scored 92.5% accuracy and 0.930 F1, versus 82.5% and 0.821 for the old rules.
The three classifier errors were false alarms; it missed none of the 20 falls.

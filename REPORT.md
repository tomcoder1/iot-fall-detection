# IoT-Based Human Fall Detection System Using Raspberry Pi 4, Coral TPU, Computer Vision, and Mobile Alerts

**Project repositories:**

- `tomcoder1/iot-fall-detection`
- `tomcoder1/fall_alert_app`

**Prepared for:** IoT Fall Detection Project  
**Group members:**

| Student ID | Full name |
|---|---|
| 104240044 | Phạm Tommy |
| 104240377 | Nguyễn Trung Kiên |

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Project Objectives](#4-project-objectives)
5. [Project Scope](#5-project-scope)
6. [Overall System Architecture](#6-overall-system-architecture)
7. [Hardware Design](#7-hardware-design)
8. [Software Design](#8-software-design)
9. [Repository Implementation Overview](#9-repository-implementation-overview)
10. [Fall Detection Method](#10-fall-detection-method)
11. [IoT Server and Mobile Application Integration](#11-iot-server-and-mobile-application-integration)
12. [Raspberry Pi Dataset, Training, and Evaluation](#12-raspberry-pi-dataset-training-and-evaluation)
13. [Deployment and Usage Instructions](#13-deployment-and-usage-instructions)
14. [Testing Plan](#14-testing-plan)
15. [Resource Management, Privacy, and Safety](#15-resource-management-privacy-and-safety)
16. [Limitations](#16-limitations)
17. [Troubleshooting](#17-troubleshooting)
18. [Future Improvements](#18-future-improvements)
19. [Conclusion](#19-conclusion)
20. [References](#20-references)

---

## 1. Abstract

This project implements an edge-based IoT fall detection system whose primary deployment target is a Raspberry Pi 4 with a Google Coral USB Accelerator. A room camera supplies frames to Coral PoseNet, which extracts 17 body keypoints. A lightweight temporal tree-ensemble classifier then evaluates 1.5 seconds of pose history, while a FastAPI server, Firebase Cloud Messaging, and a Flutter application deliver alerts and on-demand live video to a caregiver.

The final evaluation was performed on the real Raspberry Pi and Coral pipeline, rather than assuming that Windows results transfer to the edge device. Coral PoseNet processed all 160 GMDCSA24 videos and produced 34,172 frame records. Because its keypoint confidence and missing-pose behavior differed materially from Windows MoveNet, a dedicated Pi classifier was trained from Coral-extracted keypoints. The data were split randomly and stratified at video level into 120 training videos and 40 held-out test videos. Model and event-rule selection used four-fold out-of-fold predictions within the training split.

The selected Pi model is a 200-tree Extra Trees classifier with a probability threshold of 0.70 and a voting rule requiring three positive frames among the latest four. A complete second pass through the held-out videos on the physical Pi produced 20 true positives, 7 false positives, 13 true negatives, and 0 false negatives. This corresponds to 82.5% accuracy, 74.1% precision, 100% recall, 65.0% specificity, and an F1 score of 0.851. The result demonstrates complete fall recall on this split, but the false-alert rate remains an important limitation.

Windows MoveNet is retained as a secondary development and comparison platform with its own classifier artifact. The operational claim of this report, however, is based on the Raspberry Pi result. The system reports a **possible fall detected** event rather than a medically confirmed fall, allowing a caregiver to review the live feed before deciding what action is needed.

---

## 2. Introduction

Falls are a serious safety risk for elderly people, disabled users, and people with mobility problems. A fall can become more dangerous when the person is alone and cannot immediately call for help. A practical monitoring system should detect possible fall events quickly, notify a caregiver, and allow the situation to be checked without requiring continuous manual monitoring.

This project solves the problem using an edge-computing IoT design. The Raspberry Pi 4 acts as the local processing unit. It receives camera frames, runs pose estimation, applies fall detection logic, and exposes an IoT API for status, metrics, live stream control, and notification registration. The mobile application acts as the caregiver interface. It connects to the Raspberry Pi, receives status updates, registers for Firebase Cloud Messaging, and displays live video when requested.

The system is designed around three principles:

1. **Local processing first:** Camera frames are processed on the Raspberry Pi where possible instead of being continuously sent to the cloud.
2. **Temporal fall detection:** The system evaluates a short sequence of pose keypoints instead of making a decision from one frame.
3. **Human confirmation:** The mobile app allows the caregiver to open the live view after an alert so the situation can be checked before action is taken.

---

## 3. Problem Statement

A simple motion detector is not reliable enough for fall detection. Normal activities such as sitting, lying down, bending, kneeling, or moving quickly can look similar to a fall in a single frame. A camera-based fall detection system must therefore consider both body posture and movement over time.

The main problem is to build a low-cost prototype that can:

- Detect human pose from a room camera.
- Recognize fall-like movement using body keypoints.
- Avoid triggering alerts for normal activities as much as possible.
- Send alerts to a mobile device.
- Allow live video checking only when needed.
- Run on Raspberry Pi 4 hardware with limited processing power.

The system must also handle practical constraints such as camera angle, lighting, frame rate, false alerts, privacy, and mobile connectivity.

---

## 4. Project Objectives

The project objectives are:

| Objective | Description |
|---|---|
| Build a working fall detection prototype | Develop a system that can monitor a room and detect possible falls. |
| Use Raspberry Pi 4 as the edge device | Run camera capture, pose estimation, fall classification, alert handling, and IoT communication on the Pi. |
| Use computer vision for body keypoints | Detect human keypoints such as shoulders, hips, knees, and ankles. |
| Use temporal fall classification | Classify fall events using keypoint history instead of one-frame rules only. |
| Provide mobile alerts | Notify caregivers when a possible fall is detected. |
| Provide on-demand live view | Allow the caregiver to open the live camera feed after an alert. |
| Reduce unnecessary resource usage | Avoid continuous streaming and use efficient runtime logic. |
| Identify limitations | Document lighting issues, false positives, false negatives, privacy concerns, and hardware constraints. |

---

## 5. Project Scope

### 5.1 Included in the Prototype

The current project includes:

- Raspberry Pi 4 based fall detection runtime.
- Google Coral USB Accelerator support for Raspberry Pi PoseNet inference.
- Windows test runtime using MoveNet MultiPose.
- Platform-specific temporal keypoint tree-ensemble classifiers.
- Separate classifier artifacts for Windows and Raspberry Pi.
- FastAPI IoT server on the Raspberry Pi.
- REST endpoints for status, metrics, streaming, and notification registration.
- WebSocket endpoint for live status updates and fall alert events.
- Firebase Cloud Messaging support for push notifications.
- Flutter mobile app for caregiver monitoring and live view.
- On-demand MJPEG live stream.
- Local HUD and debugging output.
- Dataset-based training and held-out testing workflow.

### 5.2 Not Included in the Prototype

The current project does not fully solve every real-world deployment issue. The following are outside the current prototype scope:

- Guaranteed medical-grade fall detection.
- Night vision without external lighting.
- Full identity tracking between multiple people.
- Cloud dashboard for multiple homes or users.
- Encrypted authentication layer for all API endpoints.
- Automatic emergency service calling.
- Long-term production reliability testing.
- Robust operation in all room layouts and camera angles.

---

## 6. Overall System Architecture

The system is divided into three major parts:

1. **Camera and edge processing unit**
2. **IoT communication layer**
3. **Mobile caregiver application**

```text
+-------------------+        +----------------------------------+
| Camera / Webcam   | -----> | Raspberry Pi 4 + Coral TPU       |
| Room monitoring   |        | PoseNet + temporal classifier    |
+-------------------+        | FastAPI IoT server               |
                             | Firebase notification sender     |
                             +----------------+-----------------+
                                              |
                         REST / WebSocket / FCM / MJPEG
                                              |
                             +----------------v-----------------+
                             | Flutter Mobile Application       |
                             | Status, alerts, live view        |
                             +----------------------------------+
```

### 6.1 Normal Monitoring Flow

1. The camera captures frames from the room.
2. The Raspberry Pi receives the frames.
3. The pose estimation model detects human body keypoints.
4. The classifier receives a temporal sequence of keypoints.
5. The classifier estimates the probability of a fall.
6. The system updates the local status and mobile WebSocket status.
7. The live video stream remains inactive unless requested.

### 6.2 Alert Flow

1. The classifier triggers a fall status.
2. The IoT server creates a `possible_fall` alert event.
3. The server sends the alert through WebSocket and Firebase Cloud Messaging.
4. The mobile app updates the UI and displays the alert.
5. The caregiver can press **Open Live View**.
6. The app requests the Pi to start streaming.
7. The Pi serves the MJPEG stream through `/video_feed`.
8. The stream is stopped when the live view page is closed.

---

## 7. Hardware Design

### 7.1 Main Components

| Component | Quantity | Function |
|---|---:|---|
| Raspberry Pi 4 | 1 | Main edge processing unit for camera input, fall detection, API server, and communication. |
| Google Coral USB Accelerator | 1 | Accelerates PoseNet inference on Raspberry Pi. |
| Camera or webcam | 1 | Captures room video for pose estimation. |
| MicroSD card | 1 | Stores Raspberry Pi OS, Python environment, models, and project code. |
| USB-C power supply | 1 | Provides stable power to Raspberry Pi 4. |
| Internet or local network connection | 1 | Allows mobile app communication, live stream access, and notifications. |
| Mobile phone | 1 or more | Runs the caregiver Flutter app. |
| Camera mount or bracket | 1 | Keeps the camera fixed at a stable angle. |
| Optional cooling fan or heatsink | 1 | Helps prevent thermal throttling during continuous operation. |

### 7.2 Camera Placement

The camera should be placed at the top corner of the room. This gives a wider field of view and makes it easier to observe both vertical movement and final body posture. The camera should face the main area where the user is expected to move, such as the bed area, open floor, or walking path.

Important camera placement rules:

- Keep the camera fixed and stable.
- Avoid placing large furniture between the user and camera.
- Avoid direct backlight from windows.
- Ensure the floor area is visible.
- Test both standing and lying positions in the camera frame.

### 7.3 Lighting Requirements

The prototype does not include infrared lighting. It depends on visible light. Detection accuracy may decrease in dark rooms because pose estimation becomes less reliable when the body is not clearly visible.

Recommended lighting conditions:

- Normal room light for testing and demonstration.
- Small night lamp for low-light use.
- Avoid complete darkness.
- Avoid strong shadows or flickering light.

---

## 8. Software Design

### 8.1 Software Stack

| Layer | Technology |
|---|---|
| Edge runtime | Python |
| Camera processing | OpenCV |
| Windows pose estimation | MoveNet MultiPose TFLite |
| Raspberry Pi pose estimation | Coral PoseNet with Google Coral USB Accelerator, based on the Google Coral PoseNet Edge TPU project [4] |
| Fall classifier runtime | NumPy-based exported decision-tree ensemble evaluator |
| IoT API | FastAPI and Uvicorn |
| Live stream | MJPEG stream over HTTP |
| Status updates | WebSocket |
| Push notifications | Firebase Cloud Messaging |
| Mobile app | Flutter and Dart |
| Mobile live view | WebView displaying MJPEG stream |

### 8.2 Main Software Modules

| Module | Purpose |
|---|---|
| Camera capture module | Opens the camera, configures resolution and FPS, and reads frames. |
| Pose model module | Converts frames into 17 body keypoints. |
| Keypoint normalization module | Converts model-specific keypoint outputs into a shared format. |
| Fall classifier module | Uses temporal keypoint features to classify possible falls. |
| Runtime loop | Connects camera capture, pose estimation, classifier, display, and IoT state updates. |
| IoT server | Provides status, metrics, WebSocket, notification registration, and streaming endpoints. |
| Push notification module | Sends Firebase Cloud Messaging alerts to registered mobile devices. |
| Mobile app | Shows connection state, fall status, people count, notifications, and live view. |

---

## 9. Repository Implementation Overview

## 9.1 `iot-fall-detection`

The `iot-fall-detection` repository is the main edge detection and server repository. It contains the Python runtime for both Windows and Raspberry Pi.

Important files and folders:

| Path | Description |
|---|---|
| `main_pi.py` | Entry point for Raspberry Pi runtime. It imports and runs the Coral PoseNet detector. |
| `main_win.py` | Entry point for Windows runtime. It imports and runs the MoveNet MultiPose detector. |
| `detectors/pi4_coral_posenet_fall.py` | Raspberry Pi + Coral PoseNet implementation. |
| `detectors/windows_movenet_multipose_fall.py` | Windows MoveNet MultiPose implementation. |
| `detectors/fall_classifier.py` | Shared temporal keypoint classifier runtime. |
| `detectors/keypoint_features.py` | Feature extraction from keypoint history. |
| `app_common.py` | Shared camera loop, display HUD, pose drawing, and classifier update flow. |
| `iot_server.py` | FastAPI server, status endpoint, metrics endpoint, WebSocket, live MJPEG stream, and notification registration. |
| `push_notifications.py` | Firebase notification support. |
| `models/fall_classifier_windows.json` | Exported random forest classifier for Windows MoveNet keypoints. |
| `models/fall_classifier_pi.json` | Exported Extra Trees classifier trained from Raspberry Pi Coral PoseNet keypoints. |
| `models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite` | Edge TPU PoseNet model for Raspberry Pi. |
| `train/` | Training, keypoint extraction, model selection, and evaluation tools. |
| `test_win.py` | Held-out Windows classifier test script. |
| `test_pi.py` | Held-out Raspberry Pi classifier test script. |

### 9.2 `fall_alert_app`

The `fall_alert_app` repository contains the Flutter mobile app. It is the caregiver-facing client.

Important app responsibilities:

- Connect to the Raspberry Pi server by IP address and port.
- Listen for WebSocket status updates.
- Register the phone's Firebase Cloud Messaging token with the Pi.
- Display the current fall status.
- Display people count and disabled reason when available.
- Display the latest alert time.
- Open the live MJPEG stream in a WebView.
- Start the stream when entering the live view page.
- Stop the stream when leaving the live view page.

---

## 10. Fall Detection Method

### 10.1 Initial Rule-Based Approach

The first approach used in this project was a rule-based fall detection method. This approach was useful as an early baseline because it allowed the system to test whether pose keypoints could be used to detect fall-like movement. However, the rule-based method was not selected as the final solution because it was less reliable when normal actions looked similar to a fall, such as sitting, bending, lying down, or moving partly out of the camera frame.

The rule-based detector is retained only as a historical Windows benchmark. It is not used by the deployed Pi runtime. The final Pi design instead uses a classifier trained directly from Coral PoseNet output.

### 10.2 Final Temporal Tree-Ensemble Approach

The final fall detection method uses a temporal tree ensemble. Candidate models included Random Forest, Extra Trees, and equal-weight forest ensembles. The selected Pi model is Extra Trees, a randomized decision-tree ensemble that remains lightweight enough for CPU inference on Raspberry Pi 4.

The tree ensemble receives pose keypoints extracted from a short sequence of video frames. Each frame provides body landmark positions such as shoulders, hips, knees, and ankles. Instead of classifying one frame by itself, the classifier analyzes movement over time. This is important because a fall is not only a posture. A fall is a motion pattern: the person changes from an upright or moving state into a low or lying position within a short time.

The classifier-based approach follows this pipeline:

```text
Camera frame
   |
Coral PoseNet on Edge TPU
   |
17 body keypoints
   |
1.5-second temporal history
   |
Posture and motion-delta features
   |
Pi-specific Extra Trees classifier
   |
Fall probability and 3-of-4 voting
   |
Possible fall alert
```

### 10.3 Pose Keypoints Used by the Classifier

Both MoveNet MultiPose and Coral PoseNet produce 17 human body keypoints:

| Index | Keypoint |
|---:|---|
| 0 | Nose |
| 1 | Left eye |
| 2 | Right eye |
| 3 | Left ear |
| 4 | Right ear |
| 5 | Left shoulder |
| 6 | Right shoulder |
| 7 | Left elbow |
| 8 | Right elbow |
| 9 | Left wrist |
| 10 | Right wrist |
| 11 | Left hip |
| 12 | Right hip |
| 13 | Left knee |
| 14 | Right knee |
| 15 | Left ankle |
| 16 | Right ankle |

Each keypoint contains an `x` coordinate, a `y` coordinate, and a confidence score. These values describe the estimated position of the body in the camera frame. The system normalizes and stores recent keypoints so that movement can be analyzed across time.

### 10.4 Pi Temporal Feature Extraction

The Pi classifier does not directly use raw video pixels. Coral performs the expensive pose estimation, after which the classifier uses numeric features derived from five temporal snapshots at 1.50, 1.00, 0.50, 0.25, and 0.00 seconds before the current decision. Each snapshot contains pose presence and confidence, valid-keypoint coverage, body center, body dimensions and aspect ratio, shoulder and hip positions, torso angle, normalized coordinates for all 17 keypoints, and keypoint confidence values.

The feature extraction stage summarizes recent motion and posture information, including:

| Feature type | Purpose |
|---|---|
| Body center movement | Measures whether the person moves sharply downward or changes position quickly. |
| Shoulder and hip position | Represents upper-body posture and orientation. |
| Knee and ankle position | Helps distinguish standing, crouching, sitting, and lying states. |
| Bounding body shape | Captures whether the body appears vertical, compact, or horizontal. |
| Explicit motion deltas | Subtracts adjacent temporal snapshots so shallow trees can evaluate movement directly. |
| Keypoint confidence | Helps reduce the effect of unreliable or missing pose points. |

This design is more flexible than fixed rules. For example, a simple rule may classify a low body position as a fall, but the learned ensemble can use motion history to distinguish falling from lying down slowly.

### 10.5 Why a Tree Ensemble Was Selected

Tree ensembles are suitable for this project for several reasons:

| Reason | Explanation |
|---|---|
| Works well with structured features | The input is not an image. It is a set of numeric pose and motion features, which tree ensembles handle well. |
| Lightweight runtime | The trained model can be exported into a compact JSON format and evaluated using NumPy. |
| Good for limited datasets | Tree ensembles can perform well when the dataset is smaller than what deep learning normally requires. |
| Less manual tuning | It reduces dependency on hand-selected thresholds. |
| Interpretable behavior | They are easier to explain than a large neural network because they use decision trees over pose features. |
| Suitable for Raspberry Pi | The heavy pose estimation work is handled by MoveNet or Coral PoseNet, while the classifier itself is fast. |

The selected Pi model contains 200 Extra Trees estimators with maximum depth 8 and minimum leaf size 4. Each tree checks different feature conditions and contributes a class probability. The exported model is a 1.9 MB JSON artifact evaluated by a small NumPy runtime; scikit-learn is required for training but not on the Pi.

### 10.6 Separate Classifiers for Windows and Raspberry Pi

The project uses separate classifiers for Windows and Raspberry Pi because the pose models are different:

| Platform | Pose model | Classifier artifact |
|---|---|---|
| Windows | MoveNet MultiPose | `models/fall_classifier_windows.json` |
| Raspberry Pi 4 + Coral | Coral PoseNet | `models/fall_classifier_pi.json` |

This separation is important. MoveNet and Coral PoseNet both output 17 keypoints, but they do not produce identical coordinate behavior or confidence values. A model trained using MoveNet keypoints may not perform the same when given Coral PoseNet keypoints. Training and exporting separate classifiers makes the system more stable on each platform.

### 10.7 Classifier Output and Confirmation Logic

The classifier produces a fall probability for the recent keypoint sequence. The runtime does not immediately trigger an alert from only one positive prediction. Instead, it uses a confirmation mechanism.

The classifier output includes:

| Output | Meaning |
|---|---|
| `probability` | Estimated probability that the recent motion represents a fall. |
| `threshold` | Minimum probability required for a positive fall vote. |
| `votes` | Number of recent positive predictions. |
| `status` | Current state, such as `OK`, `POSSIBLE_FALL`, or `FALL`. |
| `triggered` | Runtime flag used to hold the alarm state. |

The Pi configuration uses a probability threshold of `0.70` and triggers when at least `3` of the latest `4` frames are positive. This short voting window tolerates one weak or missing Coral pose while still requiring repeated evidence. The Windows model uses a stricter `4`-of-`4` rule, but that configuration is secondary to the Pi evaluation in this report.

Missing-pose frames are not automatically discarded. Coral often loses a person briefly when the body reaches the floor; the earlier snapshots can still contain valid fall motion. The classifier was trained and tuned with these gaps, so a short loss of the current pose may contribute evidence through temporal history. When multiple poses are returned, extraction and runtime both follow the highest-confidence pose to maintain train/runtime consistency.

After a fall is triggered, the alarm state is held for a short period. In the current runtime configuration, the alarm hold time is 5 seconds. This keeps the alert visible long enough for the IoT server and mobile application to update.

### 10.8 Multi-Person Handling

The system can detect multiple visible people through the pose model. In the current implementation, fall detection remains active and evaluates the highest-confidence accepted pose. The IoT server and mobile app still report the number of accepted people.

This keeps the system usable when another person appears briefly in the frame. However, it also means that if multiple people are visible, the classifier may evaluate the wrong person if the wrong pose has the highest confidence. This is a limitation of the current prototype and should be improved with target-person tracking in future work.

### 10.9 Comparison Between Rule-Based and Learned Approaches

| Aspect | Rule-based approach | Learned tree-ensemble approach |
|---|---|---|
| Role in project | First approach and baseline | Final detection method |
| Input | Pose keypoints and manually selected conditions | Temporal pose and motion features |
| Decision method | Fixed thresholds | Learned decision trees |
| Tuning difficulty | High | Lower after training |
| Robustness | More sensitive to camera angle and activity variation | More flexible because it learns from examples |
| Runtime cost | Very low | Low |
| Final suitability | Useful baseline only | Selected for final prototype |

The learned approach is the stronger final method because it keeps the system lightweight while allowing the detector to learn from dataset examples. Its main advantage for this project is that the Pi model can be trained from the actual Coral keypoint distribution rather than assuming Windows thresholds or MoveNet outputs will transfer correctly.

---

## 11. IoT Server and Mobile Application Integration

### 11.1 Raspberry Pi IoT Server

The Raspberry Pi runtime starts a FastAPI server on port `8000`. This server exposes endpoints for monitoring, streaming, notifications, and mobile app communication.

| Endpoint | Method | Purpose |
|---|---|---|
| `/status` | GET | Returns current fall status, message, people count, disabled reason, latest alert, and stream state. |
| `/metrics` | GET | Returns detection FPS, stream state, CPU usage, memory usage, CPU temperature, uptime, and alert count where available. |
| `/video_feed` | GET | Provides MJPEG live video stream. |
| `/stream/status` | GET | Returns whether the stream is active and how many stream clients are connected. |
| `/stream/start` | POST | Requests live stream activation. |
| `/stream/stop` | POST | Stops requested live streaming. |
| `/notifications/register` | POST | Registers a Firebase Cloud Messaging device token. |
| `/notifications/unregister` | POST | Removes a registered device token. |
| `/notifications/status` | GET | Returns Firebase notification configuration and delivery status. |
| `/ws` | WebSocket | Sends periodic status updates and fall alert events. |

### 11.2 Alert Event Deduplication

The IoT server sends an alert when the fall status changes from false to true. It does not use a time-based cooldown. The rising-edge condition prevents the 5-second alarm hold from producing one notification per camera frame. After the status returns to false, the next independent false-to-true transition can send another alert immediately.

The alert payload includes:

```json
{
  "event_id": "YYYYMMDD-HHMMSS",
  "timestamp": "ISO timestamp",
  "message": "possible fall detected",
  "status": "possible_fall",
  "stream_url": "/video_feed"
}
```

### 11.3 Mobile App Behavior

The Flutter mobile app provides the caregiver interface. The user enters the Raspberry Pi address, for example:

```text
192.168.1.50:8000
```

After connection, the app displays:

- Connection status
- Fall status
- People count
- Disabled reason if available
- Latest alert time
- Stream status
- Notification registration status

When a fall alert is received, the app displays a warning dialog. The caregiver can close the dialog or open the live view. When live view is opened, the app sends `/stream/start` to the Pi and loads `/video_feed` in a WebView. When the page closes, the app sends `/stream/stop`.

### 11.4 Firebase Cloud Messaging

The app supports Firebase Cloud Messaging. On startup, it requests notification permission and obtains an FCM token. When connected to the Pi, it registers the token with the Pi server.

If Firebase is not configured, the app can still connect to the Pi and receive local WebSocket alerts while it is open. However, real push notifications require Firebase setup on both the Flutter app and Raspberry Pi server.

---

## 12. Raspberry Pi Dataset, Training, and Evaluation

### 12.1 Dataset Structure

The GMDCSA24 data used by the repository contain 160 videos from four subjects: 79 fall videos and 81 Activities of Daily Living (ADL) videos. Each fall CSV provides an annotated fall interval. The training loader parses these annotations so that positive frame samples represent the annotated fall period rather than labeling every frame in a fall video as positive.

| Dataset item | Count |
|---|---:|
| Subjects | 4 |
| Total videos | 160 |
| Fall videos | 79 |
| ADL videos | 81 |
| Coral-extracted frame records | 34,172 |

The video set was divided randomly and stratified with seed 42. The split was performed at video level, giving 120 training videos and 40 held-out test videos. No frame from a held-out video was used for fitting or model selection.

### 12.2 Training Workflow

The central evaluation requirement was to train from the same pose distribution used in deployment. The complete Pi workflow was therefore:

1. Run `train.extract_keypoints --platform pi` on the physical Raspberry Pi 4.
2. Decode every dataset video and run the Edge TPU PoseNet model on each frame.
3. Select the highest-confidence accepted pose, matching production behavior.
4. Store 17 keypoints, confidence values, pose score, and source FPS in `train/cache/pi/`.
5. Copy the compact Pi keypoint cache to Windows for scikit-learn training.
6. Generate posture snapshots and explicit motion-delta features from 1.5 seconds of history.
7. Fit and compare Random Forest, Extra Trees, and forest-ensemble candidates.
8. Produce four-fold video-level out-of-fold probabilities using only the 120-video training split.
9. Tune the probability threshold and temporal voting rule from those out-of-fold predictions.
10. Refit the selected model on all 120 training videos and evaluate once on the 40 held-out videos.
11. Export the model to `models/fall_classifier_pi.json` and deploy it back to the Pi.

The full Coral extraction took approximately 46 minutes. PoseNet produced an accepted pose for 86.45% of all frames. This missing-pose rate was one reason a MoveNet-trained model and strict consecutive-frame rule were unsuitable for Pi deployment.

### 12.3 Pi Model Selection and Tuning

Eight lightweight candidates were evaluated: three Random Forest configurations, three Extra Trees configurations, and two averaged forest ensembles. For each candidate, four models were trained so that every training video received an out-of-fold probability from a model that had not seen that video. Event tuning searched probability thresholds from 0.30 to 0.85 and voting windows from 4 to 12 frames.

The model-selection metric was F2, which weights recall twice as strongly as precision because missing a fall was treated as more costly than issuing a false alert. Within each model, threshold and voting parameters were tuned by F1 to prevent sensitivity from increasing without control.

The selected Pi candidate was `extra_trees_d8_l4`:

| Parameter | Selected value |
|---|---:|
| Estimators | 200 |
| Maximum tree depth | 8 |
| Minimum samples per leaf | 4 |
| Probability threshold | 0.70 |
| Voting window | 4 frames |
| Required positive votes | 3 |
| Exported artifact size | 1.9 MB |

Its out-of-fold results across the 120 training videos were:

| Metric | Value |
|---|---:|
| True positives | 59 |
| False positives | 14 |
| True negatives | 47 |
| False negatives | 0 |
| Accuracy | 0.883 |
| Precision | 0.808 |
| Recall | 1.000 |
| Specificity | 0.770 |
| F1 score | 0.894 |
| F2 score | 0.955 |

### 12.4 Held-Out Raspberry Pi Evaluation

The final test was executed on the physical Pi using `test_pi.py`, the Coral USB Accelerator, the Edge TPU PoseNet model, and the 40 held-out raw videos. This was a complete second inference pass from video pixels; it did not replay cached classifier probabilities. The test therefore exercised video decoding, Coral pose inference, top-pose selection, temporal feature generation, JSON tree inference, and event voting together.

The final confusion matrix was:

| Actual / predicted | Fall | No fall |
|---|---:|---:|
| Fall | **20 TP** | **0 FN** |
| ADL | **7 FP** | **13 TN** |

| Metric | Value |
|---|---:|
| Accuracy | 0.825 |
| Precision | 0.741 |
| Recall / sensitivity | **1.000** |
| Specificity | 0.650 |
| F1 score | 0.851 |
| F2 score | 0.935 |

All 20 held-out fall videos triggered an event. Seven of the 20 held-out ADL videos also triggered, giving a 35% false-positive rate on the ADL subset. The result meets the project preference for sensitivity, but it does not establish acceptable production specificity.

### 12.5 Train/Runtime Alignment Findings

Several implementation findings were necessary to obtain the final Pi result:

- **Separate platform models were required.** On Subject 1, MoveNet produced an accepted pose in approximately 99.9% of frames, while Coral coverage was about 85.0%. Across the complete Pi cache, Coral coverage was 86.45%.
- **Strict consecutive confirmation was unsuitable.** A fall could receive a high probability and still fail because one weak Coral frame reset the counter. Validation selected a 3-of-4 voting rule instead.
- **Pose loss near the floor was meaningful.** Coral sometimes lost the current pose after the person reached the floor. Temporal history was therefore allowed to vote during short missing-pose periods, matching the data used for tuning.
- **Pose selection had to match extraction.** Extraction stored the highest-confidence pose, while an early test path discarded multi-pose frames. Using the highest-confidence pose in both training and runtime removed this mismatch and reproduced the cached evaluation exactly.
- **Explicit motion deltas improved sensitivity.** Adding differences between adjacent pose snapshots improved held-out Pi recall from 90% to 100% in the selected safety-oriented model.

### 12.6 Windows Result as a Secondary Baseline

Windows remains useful for development because MoveNet inference is faster to iterate on. Its separate feature-v1 random forest obtained TP=20, FP=3, TN=17, and FN=0 on the same held-out video identities, corresponding to 92.5% accuracy and 0.930 F1. These values must not be presented as Pi performance because they use a different pose estimator and keypoint distribution.

### 12.7 Evaluation Interpretation

The most important result is that the deployed Pi pipeline, not only the Windows prototype, detected every held-out fall in the selected split. The result also shows the cost of that decision: precision and specificity are substantially lower than recall. In practical terms, the prototype is more likely to inconvenience a caregiver with an unnecessary **possible fall** alert than to remain silent on a dataset fall.

This tradeoff is acceptable for demonstrating the safety-oriented prototype, especially because the mobile app provides live visual confirmation, but it is not sufficient for a medical or unattended commercial system. The next evaluation priority is to collect Pi-specific hard-negative examples—particularly sitting, deliberate lying down, partial occlusion, and leaving the camera view—from the intended room and camera angle. Thresholds should not be changed using the held-out set; improvements should be selected through new validation data and then assessed on a fresh test set.

---

## 13. Deployment and Usage Instructions

### 13.1 Raspberry Pi Setup

Basic setup steps:

1. Install Raspberry Pi OS.
2. Connect the camera or webcam.
3. Connect the Google Coral USB Accelerator.
4. Install Python 3.9.
5. Install required OS packages and Python packages from the project requirements.
6. Clone the repository:

```bash
git clone https://github.com/tomcoder1/iot-fall-detection.git
cd iot-fall-detection
```

7. Clone Google's PoseNet helper inside the project folder:

```bash
git clone https://github.com/google-coral/project-posenet.git
```

8. Activate the Python 3.9 Pi environment and run the detector from the repository root:

```bash
source .venv-pi/bin/activate
python --version
python main_pi.py
```

The version check should report Python 3.9.x. The Pi runtime loads `models/fall_classifier_pi.json`, uses the Edge TPU PoseNet model stored under `models/`, and starts the IoT server on port `8000`.

To repeat the 40-video held-out Pi evaluation:

```bash
python test_pi.py
```

Use `python test_pi.py --all` only when intentionally evaluating all 160 videos, including videos used for training.

### 13.2 Windows Test Setup

From the repository root:

```powershell
uv sync
uv run python main_win.py
```

To run the held-out Windows evaluation:

```powershell
uv run python test_win.py
```

Press `q` or Escape to stop the live Windows detector window.

### 13.3 Firebase Setup

For real push notifications:

1. Create a Firebase project.
2. Enable Cloud Messaging.
3. Configure the Flutter app using FlutterFire CLI:

```bash
flutterfire configure
```

4. Add the Android or iOS app to Firebase using the real package or bundle identifier.
5. Place the Firebase service account JSON on the Raspberry Pi outside source control.
6. Set the environment variable on the Pi:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/home/pi/iot-fall-detection/firebase-service-account.json
```

If Firebase is not configured, WebSocket alerts can still work while the app is connected, but push notifications will not be fully available.

### 13.4 Mobile App Setup

Basic setup steps:

```bash
git clone https://github.com/tomcoder1/fall_alert_app.git
cd fall_alert_app
flutter pub get
flutter run
```

After launching the app:

1. Enter the Raspberry Pi address and port.
2. Press **Connect**.
3. Confirm that the connection status changes to connected.
4. Confirm that notification registration is successful if Firebase is configured.
5. Press **Open Live View** to verify that the MJPEG stream works.

---

## 14. Testing Plan

### 14.1 Camera Test

| Test | Expected result |
|---|---|
| Start the Pi runtime | Camera opens successfully. |
| Stand in the camera view | Person is detected. |
| Move across the room | Pose keypoints remain stable enough for classification. |
| Lie down in view | Body remains visible in the final position. |
| Change camera angle | System behavior is checked for stability. |

### 14.2 Fall Detection Test

| Activity | Expected behavior |
|---|---|
| Normal walking | No alert. |
| Sitting on a chair | No alert. |
| Bending down | No alert. |
| Picking up an object | No alert. |
| Lying down slowly | Preferably no alert. |
| Simulated fast fall | Alert should trigger. |
| Fall followed by remaining down | Alert should trigger and hold. |
| Person leaves camera view | Should not trigger without fall-like evidence. |
| Multiple people visible | System should continue but may evaluate the highest-confidence pose. |

### 14.3 Mobile App Test

| Test | Expected result |
|---|---|
| Connect to Pi IP address | App shows connected status. |
| Check `/status` | App updates fall status and people count. |
| Trigger simulated fall | App receives WebSocket alert. |
| Firebase configured | App receives push notification. |
| Open live view | App starts stream and displays video. |
| Close live view | App stops requested stream. |

### 14.4 Resource Test

| Metric | Reason |
|---|---|
| Detection FPS | Confirms real-time or near-real-time operation. |
| CPU usage | Detects overload on Raspberry Pi. |
| Memory usage | Detects memory leaks or excessive buffering. |
| CPU temperature | Detects thermal throttling risk. |
| Stream active clients | Confirms stream is only active when needed. |
| Alert count | Confirms one alert is created per false-to-true fall transition. |

### 14.5 Evaluation Metrics

The final system should be evaluated using:

| Metric | Meaning |
|---|---|
| Accuracy | Overall percentage of correct video classifications. |
| Precision | Percentage of predicted falls that were real falls. |
| Recall | Percentage of real falls detected. |
| Specificity | Percentage of non-fall videos correctly rejected. |
| F1 score | Balance between precision and recall. |
| Alert delay | Time from fall event to alert. |
| FPS | Average detection speed. |
| CPU temperature | Stability during long use. |
| Live stream latency | Delay between real movement and mobile view. |
| Notification success rate | Percentage of alerts received by the phone. |

---

## 15. Resource Management, Privacy, and Safety

### 15.1 Resource Management

The Raspberry Pi 4 has limited CPU and thermal headroom. The project reduces unnecessary load using several design choices:

- Pose estimation runs locally instead of continuous cloud upload.
- Coral TPU accelerates the Pi pose model.
- The fall classifier is a compact NumPy-based tree-ensemble runtime.
- Live stream JPEG encoding is done only when a stream client is active.
- The stream is started and stopped through mobile app requests.
- Rising-edge alert generation prevents repeated notifications during one held alarm.
- The Pi exposes metrics for CPU, memory, temperature, uptime, stream state, and detection FPS.

### 15.2 Privacy

The system uses a camera in a private room, so privacy is a major concern. The design reduces privacy risk by:

- Processing fall detection locally on the Raspberry Pi.
- Avoiding continuous cloud video upload.
- Avoiding continuous video recording in the prototype.
- Starting the live stream only when requested.
- Using the live view mainly for post-alert confirmation.

For real deployment, additional privacy features should be added, including authentication, encryption, user consent, visible camera status, access logging, and possibly privacy-preserving blurred live view modes.

### 15.3 Safety

The system should not be described as a medical device. It is a prototype that detects possible falls. It can miss real falls or trigger false alerts. The phrase **possible fall detected** is more accurate than **fall confirmed**.

A caregiver should use the alert as a prompt to check the situation. The system should not be the only safety mechanism for high-risk users.

---

## 16. Limitations

### 16.1 Lighting Limitation

The prototype does not include infrared lighting. Detection may be poor in dark rooms or rooms with strong shadows. Pose estimation needs a visible body to produce reliable keypoints.

### 16.2 Camera Angle Limitation

Fall detection depends strongly on camera angle. A top-corner camera is recommended, but the exact angle still affects body shape, keypoint visibility, and floor position. A bad camera angle can cause false positives or missed falls.

### 16.3 Occlusion Limitation

The system may fail if the body is hidden by furniture, blankets, another person, or the bed. Keypoint detectors can lose important joints when the person is partly blocked.

### 16.4 Multi-Person Limitation

The current configuration keeps detection active and follows the highest-confidence pose when multiple people are visible. This matches training but does not guarantee that the monitored person is selected. Identity-aware tracking is required for a robust multi-person deployment.

### 16.5 Dataset Limitation

The held-out set contains only 40 videos from the same four-subject dataset used to construct the training split. Although the split is video-level, it is not subject-independent. The 100% fall recall therefore has wide uncertainty and may not transfer to elderly users, new rooms, different clothing, unusual camera angles, or low-light conditions.

The Pi test also produced 7 false alerts among 20 ADL videos. This 65% specificity is the clearest measured weakness of the current classifier and should not be hidden by reporting recall alone.

### 16.6 Notification Limitation

Push notifications depend on Firebase configuration, mobile permissions, internet access, and phone settings. WebSocket alerts require the app to be connected to the Pi.

### 16.7 Not Medical Grade

The system is a prototype and should not be treated as a certified medical fall detection system.

---

## 17. Troubleshooting

### 17.1 Camera Not Detected

Possible fixes:

- Check the camera or USB connection.
- Confirm the camera index is correct.
- Test the camera with a simple OpenCV script.
- Reboot the Raspberry Pi.
- Check whether another process is already using the camera.

### 17.2 Coral PoseNet Does Not Start

Possible fixes:

- Confirm that the Coral USB Accelerator is connected.
- Confirm that Edge TPU runtime is installed.
- Confirm that `project-posenet` exists in the repository root.
- Confirm that `pose_engine.py` exists in `project-posenet`.
- Confirm that the Edge TPU model exists under `models/`.
- Run the script from the repository root.

### 17.3 Poor Detection Accuracy

Possible fixes:

- Improve room lighting.
- Move the camera higher or to a better corner.
- Ensure the full body is visible.
- Reduce objects blocking the camera.
- Test with different fall and non-fall movements.
- Re-extract Pi keypoints and retrain the Pi classifier if needed.

### 17.4 False Alerts

Possible fixes:

- Review false-positive videos or real scenes.
- Add similar non-fall actions to the training set.
- Increase classifier threshold or required votes if needed.
- Add a bed or sofa region rule if most false alerts happen during normal lying down.
- Improve camera angle to separate sitting, lying, and falling motion.

### 17.5 Missed Falls

Possible fixes:

- Improve lighting and visibility near the floor.
- Lower the classifier threshold only after checking false positives.
- Add more fall examples to the training set.
- Confirm that the Pi classifier was trained using Pi-extracted PoseNet keypoints, not Windows MoveNet keypoints.
- Make sure the person remains visible after the fall.

### 17.6 Mobile App Cannot Connect

Possible fixes:

- Confirm the phone and Pi are on the same network.
- Confirm the Pi server is running on port `8000`.
- Open `http://<pi-ip>:8000/status` in a browser.
- Check firewall or router isolation settings.
- Confirm the app address field does not include an incorrect port.

### 17.7 Notifications Not Working

Possible fixes:

- Confirm Firebase Cloud Messaging is enabled.
- Confirm FlutterFire configuration is generated.
- Confirm Android or iOS app ID matches Firebase.
- Confirm mobile notification permission is enabled.
- Confirm `GOOGLE_APPLICATION_CREDENTIALS` is set on the Pi.
- Check `/notifications/status` on the Pi.

### 17.8 Raspberry Pi Overheating

Possible fixes:

- Add heatsink or cooling fan.
- Disable unnecessary display drawing.
- Keep live streaming off unless needed.
- Reduce camera FPS or resolution if necessary.
- Place the Pi in a ventilated area.

---

## 18. Future Improvements

Recommended improvements:

1. **Add authentication**  
   Protect `/video_feed`, `/stream/start`, `/stream/stop`, and notification endpoints with a token or login system.

2. **Add HTTPS or secure tunnel support**  
   Improve privacy and network security for remote viewing.

3. **Improve multi-person tracking**  
   Track the target user across frames instead of only using the highest-confidence pose.

4. **Add night monitoring**  
   Add infrared lighting or low-light camera support.

5. **Improve dataset diversity**  
   Add videos from more rooms, camera angles, users, clothing styles, and lighting conditions.

6. **Add sensor fusion**  
   Combine camera detection with wearable IMU, floor vibration, bed pressure sensor, or mmWave radar.

7. **Add automatic event snapshots**  
   Save a small privacy-controlled snapshot or short clip around the alert for review, only with consent.

8. **Improve mobile app UI**  
   Add alert history, caregiver contact options, acknowledgement button, and device health status.

9. **Add watchdog and service startup**  
   Run the Pi detector as a systemd service with automatic restart after failure.

10. **Add long-duration testing**  
    Test the system for several hours or days to measure stability, heat, memory usage, and false alerts.

---

## 19. Conclusion

This project produced and tested a complete Raspberry Pi 4 fall-alert pipeline: camera video is processed by Coral PoseNet on the Edge TPU, recent keypoints are evaluated by a Pi-specific temporal Extra Trees classifier, and possible-fall events are delivered through the local IoT server, Firebase, and the Flutter caregiver application.

The main contribution is the Pi-specific evaluation process. All 160 videos were processed by the real Coral device, generating 34,172 frame records with 86.45% pose coverage. Model selection used four-fold out-of-fold predictions inside a 120-video training split, and final evaluation used a separate 40-video test split. A complete raw-video test on the Pi obtained TP=20, FP=7, TN=13, and FN=0: 100% recall and 0.851 F1. Windows MoveNet results were kept only as a secondary baseline and were not substituted for edge-device evidence.

The result supports the feasibility of lightweight temporal classification on Raspberry Pi, but it also defines the present limitation clearly. Detecting all held-out falls required accepting seven false alerts, and the dataset contains only four subjects. The prototype must therefore be described as a **possible-fall alert and caregiver confirmation system**, not a medical fall-confirmation device.

The next technical priority is not simply lowering the threshold. It is collecting representative Pi/Coral hard negatives and new subject-independent test data from the intended room, followed by validation-controlled retraining. Alongside stronger authentication, low-light support, target-person tracking, and long-duration testing, this would provide more meaningful evidence of real-world reliability.

---

## 20. References

1. Google Coral. (2023). *Coral PoseNet: Human Pose Detection on Edge TPU*. GitHub. <https://github.com/google-coral/project-posenet>
2. Alam, E., Sufian, A., Dutta, P., Leo, M., & Hameed, I. A. (2024). *GMDCSA24: A Dataset for Human Fall Detection in Videos*. Data in Brief. Repository: <https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos>
3. Alam, E. (2024). *ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos: 2.0* [Dataset]. Zenodo. <https://doi.org/10.5281/zenodo.12921216>
4. Subburam, R., Chandralekha, E., & Kandasamy, V. (2023). *An Elderly Fall Detection System Using Enhanced Random Forest in Machine Learning*. Engineering Proceedings, 59(1), 172. <https://doi.org/10.3390/engproc2023059172>

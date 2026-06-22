# IoT fall detection

## What you need

- Raspberry Pi 4 with Raspberry Pi OS Bullseye (Legacy)
- Google Coral USB Accelerator
- USB webcam or another camera exposed as a `/dev/video*` device
- Phone and Pi connected to the same local network
- A second computer with Flutter and Android Studio for building the mobile app
- Optional: a Firebase project for push notifications when the app is closed

The Pi runtime requires **CPython 3.9** because the Coral Python wheels used by
this project are pinned to that version. Bullseye includes Python 3.9. Do not use
the normal Bookworm image unless you intend to build and maintain a separate
Python 3.9 environment yourself.

## 1. Prepare the Raspberry Pi

Flash Raspberry Pi OS Bullseye (Legacy), boot the Pi, connect it to the network,
and update it:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo reboot
```

Connect the webcam and Coral USB Accelerator after the Pi has restarted. Confirm
that the camera is visible:

```bash
ls /dev/video*
```

Install Git, camera utilities, Python build tools, OpenCV, and NumPy:

```bash
sudo apt update
sudo apt install -y \
  git curl gpg build-essential v4l-utils \
  python3-pip \
  python3-numpy
```

### Install the Coral Edge TPU runtime

Add Google's Coral package repository and install the standard-speed runtime:

```bash
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg

echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
  | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list

sudo apt update
sudo apt install -y libedgetpu1-std
```

Unplug and reconnect the Coral after installation.

### Download and install this project

```bash
git clone https://github.com/tomcoder1/iot-fall-detection.git
cd iot-fall-detection
git clone https://github.com/google-coral/project-posenet.git

python3.9 -m venv --system-site-packages .venv-pi
source .venv-pi/bin/activate
python -m pip install --upgrade "pip<26" setuptools wheel
python -m pip install -r pi_requirements.txt
python -m pip install "numpy<2" opencv-python
```

Verify the installation from the repository root:

```bash
python --version
python -c "import cv2, numpy; print('OpenCV', cv2.__version__, 'NumPy', numpy.__version__)"
python -c "import pycoral, tflite_runtime; print('Coral Python OK')"
python -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
test -e "project-posenet/posenet_lib/$(uname -m)/posenet_decoder.so" && echo "PoseNet decoder OK"
```

`python --version` must report Python 3.9.x, and `list_edge_tpus()` should list
the connected accelerator. On 32-bit Pi OS, if the PoseNet decoder check fails
because an `armv7l` link was copied incorrectly, repair it with:

```bash
rm -f project-posenet/posenet_lib/armv7l
ln -s armv7a project-posenet/posenet_lib/armv7l
```

## 2. Configure and run the detector

If the Pi runs over SSH or without a desktop, change:

```python
DISPLAY = False
```

If the wrong camera opens, change `CAMERA_INDEX`. You can inspect camera device
names with:

```bash
v4l2-ctl --list-devices
```

Start the detector from the repository root:

```bash
cd ~/iot-fall-detection
source .venv-pi/bin/activate
python main_pi.py
```

Keep this process running while using the phone app. Find the Pi address with:

```bash
hostname -I
```

From another device on the same network, open the following URL to confirm that
the server is reachable, replacing the address with the Pi's IP:

```text
http://<PI-IP>:8000/status
```

The server also provides `/metrics`, `/video_feed`, `/stream/status`, notification
registration endpoints, and a WebSocket at `ws://<pi-ip>:8000/ws`.

## 3. Set up the mobile app

The Flutter client is maintained in a separate repository. Install the current
stable Flutter SDK and Android Studio on your development computer, install the
Android SDK when prompted, then check the toolchain:

```bash
flutter doctor
flutter doctor --android-licenses
```

Enable Developer options and USB debugging on an Android phone, connect it by
USB, and confirm Flutter can see it:

```bash
flutter devices
```

Download and run the app:

```bash
git clone https://github.com/tomcoder1/fall_alert_app.git
cd fall_alert_app
flutter pub get
flutter run
```

The phone and Pi must be on the same Wi-Fi/LAN. In the app:

1. Enter `<pi-ip>:8000`.
2. Tap **Connect** and check that the connection changes to **Connected**.
3. Tap **Check Status** to verify fall state and people count.
4. Tap **Open Live View** to verify the MJPEG camera stream.

The default `raspberrypi.local:8000` address may work through mDNS. Use the
numeric IP from `hostname -I` if the phone cannot resolve it. If connection still
fails, ensure the detector is running, both devices are on the same network, and
the router is not using client/AP isolation.

### Optional: enable Firebase push notifications

Without Firebase, the app still receives alerts over WebSocket while connected.
Firebase Cloud Messaging is required for system push notifications when the app
is in the background or closed.

Install Node.js first, then install the Firebase CLI and FlutterFire CLI. In the
mobile app directory, sign in and configure the Android and/or iOS app:

```bash
npm install --global firebase-tools
firebase login
dart pub global activate flutterfire_cli
flutterfire configure
flutter pub get
```

Choose the same Firebase project for the phone and Pi. `flutterfire configure`
generates `lib/firebase_options.dart` and the platform Firebase configuration.
For a production app, replace the example Android application ID and iOS bundle
ID before configuring Firebase.

In Firebase Console, open **Project settings > Service accounts**, generate a
private key, and copy the downloaded JSON to the Pi. Keep it private and outside
Git. For example:

```bash
mkdir -p ~/.config/fall-detection
chmod 700 ~/.config/fall-detection
# Copy the downloaded JSON to this directory as firebase-service-account.json.
chmod 600 ~/.config/fall-detection/firebase-service-account.json

export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/fall-detection/firebase-service-account.json"
cd ~/iot-fall-detection
source .venv-pi/bin/activate
python main_pi.py
```

The environment variable must be set in the same shell that starts the detector.
After granting notification permission on the phone, connect the app to the Pi.
The app's **Notifications** row should change to **Registered**. You can inspect
the Pi-side status at:

```text
http://<pi-ip>:8000/notifications/status
```

For iOS, run the app from macOS with Xcode, enable the Push Notifications and
Background Modes > Remote notifications capabilities, and upload an APNs key to
the Firebase project.

## Troubleshooting

- **Camera does not open:** check `v4l2-ctl --list-devices`, close other programs
  using the camera, and set the correct `CAMERA_INDEX`.
- **`Could not connect to Edge TPU`:** reconnect the Coral, confirm
  `libedgetpu1-std` is installed, and rerun the `list_edge_tpus()` check.
- **PoseNet decoder is missing:** ensure `project-posenet` was cloned directly
  inside this repository and run from the repository root.
- **OpenCV fails over SSH:** set `DISPLAY = False`.
- **Phone cannot connect:** first load `http://<pi-ip>:8000/status` in the phone's
  browser; check the address, Wi-Fi network, firewall, and router isolation.
- **Notifications say Pi Firebase is not configured:** check
  `GOOGLE_APPLICATION_CREDENTIALS`, the JSON file permissions, and
  `/notifications/status`.

## Development and evaluation

Windows development uses a separate MoveNet pose model and classifier:

```powershell
uv sync
uv run python main_win.py
uv run python test_win.py
```
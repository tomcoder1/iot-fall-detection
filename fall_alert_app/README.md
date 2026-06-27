# Fall alert app

Flutter client for the Raspberry Pi fall detector. It receives foreground
status updates over WebSocket, registers the phone for Firebase Cloud Messaging,
and displays the on-demand MJPEG live view.

## Firebase setup

1. Create a Firebase project and enable Cloud Messaging.
2. Install the FlutterFire CLI and run `flutterfire configure` in this folder.
3. Add the Android and/or iOS app in Firebase. Use the app's real package or
   bundle identifier rather than `com.example` for a production build.
4. On iOS, enable Push Notifications and Background Modes > Remote notifications
   in Xcode, then upload an APNs key in Firebase.
5. Run the app once, connect it to the Pi, and verify that the Notifications row
   changes to `Registered`.

The Pi needs the matching Firebase service-account credentials described in the
detector project's README. Without Firebase configuration the app still runs,
but the Notifications row remains `Not configured` and only local WebSocket
alerts are available.

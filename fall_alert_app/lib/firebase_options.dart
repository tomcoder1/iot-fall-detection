// Placeholder that keeps the app buildable before a Firebase project is chosen.
// `flutterfire configure` replaces this file with project-specific values.
import 'package:firebase_core/firebase_core.dart' show FirebaseOptions;
import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform, kIsWeb;

abstract final class DefaultFirebaseOptions {
  static FirebaseOptions get currentPlatform {
    if (kIsWeb) return web;
    switch (defaultTargetPlatform) {
      case TargetPlatform.android:
        return android;
      case TargetPlatform.iOS:
        return ios;
      default:
        throw UnsupportedError(
          'Firebase Messaging is only configured for Android, iOS, and web.',
        );
    }
  }

  static const FirebaseOptions android = FirebaseOptions(
    apiKey: 'AIzaSyCoff8Ph-sB0OxQyxb4NPnzYHTGI5FOric',
    appId: '1:197094605716:android:72ed8f0f5625a3ef044581',
    messagingSenderId: '197094605716',
    projectId: 'fall-alert-1e191',
    storageBucket: 'fall-alert-1e191.firebasestorage.app',
  );
  static const FirebaseOptions ios = FirebaseOptions(
    apiKey: '',
    appId: '',
    messagingSenderId: '',
    projectId: '',
    iosBundleId: 'com.example.fallAlertApp',
  );

  static const FirebaseOptions web = FirebaseOptions(
    apiKey: '',
    appId: '',
    messagingSenderId: '',
    projectId: '',
  );
}

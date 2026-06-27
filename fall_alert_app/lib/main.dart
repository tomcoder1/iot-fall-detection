import 'dart:async';
import 'dart:convert';

import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';
import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:http/http.dart' as http;
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:webview_flutter/webview_flutter.dart';

import 'config.dart';
import 'firebase_options.dart';

const AndroidNotificationChannel _fallAlertChannel = AndroidNotificationChannel(
  'fall_alerts',
  'Fall alerts',
  description: 'Urgent notifications when a possible fall is detected.',
  importance: Importance.max,
);

final FlutterLocalNotificationsPlugin _localNotifications =
    FlutterLocalNotificationsPlugin();

@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
}

Future<bool> _initializeNotifications() async {
  try {
    await Firebase.initializeApp(
      options: DefaultFirebaseOptions.currentPlatform,
    );
    FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
    await _localNotifications.initialize(
      settings: const InitializationSettings(
        android: AndroidInitializationSettings('@mipmap/ic_launcher'),
        iOS: DarwinInitializationSettings(),
      ),
    );
    await _localNotifications
        .resolvePlatformSpecificImplementation<
          AndroidFlutterLocalNotificationsPlugin
        >()
        ?.createNotificationChannel(_fallAlertChannel);
    await FirebaseMessaging.instance.requestPermission(
      alert: true,
      badge: true,
      sound: true,
    );
    return true;
  } catch (error) {
    debugPrint('Firebase notifications are not configured: $error');
    return false;
  }
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final notificationsReady = await _initializeNotifications();
  runApp(FallAlertApp(notificationsReady: notificationsReady));
}

class FallAlertApp extends StatelessWidget {
  const FallAlertApp({required this.notificationsReady, super.key});

  final bool notificationsReady;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Fall Detection Monitor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: FallHomePage(notificationsReady: notificationsReady),
    );
  }
}

class FallHomePage extends StatefulWidget {
  const FallHomePage({required this.notificationsReady, super.key});

  final bool notificationsReady;

  @override
  State<FallHomePage> createState() => _FallHomePageState();
}

class _FallHomePageState extends State<FallHomePage> {
  final TextEditingController _addressController = TextEditingController(
    text: piAddress,
  );
  WebSocketChannel? _channel;
  StreamSubscription<dynamic>? _socketSubscription;
  StreamSubscription<String>? _tokenSubscription;
  StreamSubscription<RemoteMessage>? _messageSubscription;

  String _connectionStatus = 'Disconnected';
  bool _fallDetected = false;
  String _message = 'normal';
  int _peopleCount = 0;
  String? _disabledReason;
  String? _latestAlertTime;
  bool _streamActive = false;
  String? _notificationToken;
  String _notificationStatus = 'Not configured';

  String get _address => _addressController.text
      .trim()
      .replaceFirst(RegExp(r'^https?://'), '')
      .replaceFirst(RegExp(r'/$'), '');
  String get _httpHost => 'http://$_address';
  String get _socketUrl => 'ws://$_address/ws';

  @override
  void initState() {
    super.initState();
    _setupPushNotifications();
  }

  Future<void> _setupPushNotifications() async {
    if (!widget.notificationsReady) return;
    try {
      _notificationToken = await FirebaseMessaging.instance.getToken();
      if (mounted) {
        setState(() {
          _notificationStatus = _notificationToken == null
              ? 'Token unavailable'
              : 'Ready to register';
        });
      }
      _tokenSubscription = FirebaseMessaging.instance.onTokenRefresh.listen((
        token,
      ) {
        _notificationToken = token;
        if (_connectionStatus == 'Connected') {
          unawaited(_registerNotificationToken());
        }
      });
      _messageSubscription = FirebaseMessaging.onMessage.listen(
        _handleForegroundPush,
      );

      final initialMessage = await FirebaseMessaging.instance
          .getInitialMessage();
      if (initialMessage != null) _applyPushData(initialMessage.data);
    } catch (_) {
      if (mounted) setState(() => _notificationStatus = 'Unavailable');
    }
  }

  Future<void> _handleForegroundPush(RemoteMessage message) async {
    _applyPushData(message.data);
    final notification = message.notification;
    await _localNotifications.show(
      id: message.messageId.hashCode,
      title: notification?.title ?? 'Fall alert',
      body: notification?.body ?? 'possible fall detected',
      notificationDetails: const NotificationDetails(
        android: AndroidNotificationDetails(
          'fall_alerts',
          'Fall alerts',
          channelDescription:
              'Urgent notifications when a possible fall is detected.',
          importance: Importance.max,
          priority: Priority.high,
        ),
        iOS: DarwinNotificationDetails(
          presentAlert: true,
          presentBadge: true,
          presentSound: true,
        ),
      ),
    );
  }

  void _applyPushData(Map<String, dynamic> data) {
    if (!mounted) return;
    setState(() {
      _fallDetected = true;
      _message = data['message']?.toString() ?? 'possible fall detected';
      _latestAlertTime = data['timestamp']?.toString() ?? _latestAlertTime;
    });
  }

  Future<void> _registerNotificationToken() async {
    final token = _notificationToken;
    if (token == null) return;
    try {
      final response = await http
          .post(
            Uri.parse('$_httpHost/notifications/register'),
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode({'token': token}),
          )
          .timeout(const Duration(seconds: 5));
      if (response.statusCode != 200) {
        throw Exception('HTTP ${response.statusCode}');
      }
      final result = jsonDecode(response.body) as Map<String, dynamic>;
      final delivery = result['delivery'];
      final configured = delivery is Map<String, dynamic>
          ? delivery['configured'] == true
          : false;
      if (mounted) {
        setState(
          () => _notificationStatus = configured
              ? 'Registered'
              : 'Pi Firebase not configured',
        );
      }
    } catch (_) {
      if (mounted) setState(() => _notificationStatus = 'Registration failed');
    }
  }

  Future<void> _connect() async {
    await _disconnect();
    if (!mounted) return;
    setState(() => _connectionStatus = 'Connecting...');

    try {
      final channel = WebSocketChannel.connect(Uri.parse(_socketUrl));
      await channel.ready.timeout(const Duration(seconds: 5));
      if (!mounted) {
        await channel.sink.close();
        return;
      }
      _channel = channel;
      _socketSubscription = channel.stream.listen(
        _handleSocketMessage,
        onError: (_) => _markDisconnected('Pi not reachable'),
        onDone: () => _markDisconnected('Disconnected'),
      );
      setState(() => _connectionStatus = 'Connected');
      await _checkStatus(showFailure: false);
      await _registerNotificationToken();
    } catch (_) {
      _markDisconnected('Pi not reachable');
    }
  }

  Future<void> _disconnect() async {
    await _socketSubscription?.cancel();
    _socketSubscription = null;
    await _channel?.sink.close();
    _channel = null;
  }

  void _markDisconnected(String value) {
    if (!mounted) return;
    setState(() => _connectionStatus = value);
  }

  void _handleSocketMessage(dynamic message) {
    try {
      final data = jsonDecode(message.toString()) as Map<String, dynamic>;
      switch (data['type']) {
        case 'status_update':
          _applyStatus(data);
          break;
        case 'fall_alert':
          _latestAlertTime = data['timestamp']?.toString();
          if (mounted) {
            setState(() {
              _fallDetected = true;
              _message = 'possible fall detected';
            });
            _showFallAlert(data);
          }
          break;
      }
    } catch (_) {
      _markDisconnected('Invalid response from Pi');
    }
  }

  void _applyStatus(Map<String, dynamic> data) {
    if (!mounted) return;
    final alert = data['latest_alert'];
    setState(() {
      _connectionStatus = 'Connected';
      _fallDetected = data['fall_detected'] == true;
      _message = data['message']?.toString() ?? 'normal';
      _peopleCount = (data['people_count'] as num?)?.toInt() ?? 0;
      _disabledReason = data['disabled_reason']?.toString();
      _streamActive = data['stream_active'] == true;
      if (alert is Map<String, dynamic>) {
        _latestAlertTime = alert['timestamp']?.toString();
      }
    });
  }

  Future<void> _checkStatus({bool showFailure = true}) async {
    try {
      final response = await http
          .get(Uri.parse('$_httpHost/status'))
          .timeout(const Duration(seconds: 5));
      if (response.statusCode != 200) {
        throw Exception('HTTP ${response.statusCode}');
      }
      _applyStatus(jsonDecode(response.body) as Map<String, dynamic>);
    } catch (_) {
      if (showFailure) _markDisconnected('Pi not reachable');
    }
  }

  Future<void> _showFallAlert(Map<String, dynamic> event) async {
    await showDialog<void>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        icon: const Icon(
          Icons.warning_amber_rounded,
          color: Colors.red,
          size: 44,
        ),
        title: const Text('Fall alert'),
        content: Text(
          'possible fall detected\n\nEvent time: '
          '${event['timestamp'] ?? 'unknown'}',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(dialogContext),
            child: const Text('Close'),
          ),
          FilledButton(
            onPressed: () {
              Navigator.pop(dialogContext);
              _openLiveView();
            },
            child: const Text('Open Live View'),
          ),
        ],
      ),
    );
  }

  Future<void> _openLiveView() async {
    await Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => LiveViewPage(host: _httpHost)),
    );
    await _checkStatus(showFailure: false);
  }

  @override
  void dispose() {
    _socketSubscription?.cancel();
    _tokenSubscription?.cancel();
    _messageSubscription?.cancel();
    _channel?.sink.close();
    _addressController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final alertColor = _fallDetected ? Colors.red : Colors.green;
    return Scaffold(
      appBar: AppBar(title: const Text('Fall Detection Monitor')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _addressController,
            keyboardType: TextInputType.url,
            decoration: const InputDecoration(
              labelText: 'Pi address',
              hintText: '',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              FilledButton(onPressed: _connect, child: const Text('Connect')),
              OutlinedButton(
                onPressed: _checkStatus,
                child: const Text('Check Status'),
              ),
              OutlinedButton.icon(
                onPressed: _openLiveView,
                icon: const Icon(Icons.videocam),
                label: const Text('Open Live View'),
              ),
            ],
          ),
          const SizedBox(height: 20),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _StatusRow('Connection', _connectionStatus),
                  _StatusRow(
                    'Fall status',
                    _fallDetected ? 'POSSIBLE FALL' : _message,
                    valueColor: alertColor,
                  ),
                  _StatusRow('People count', '$_peopleCount'),
                  if (_disabledReason != null)
                    _StatusRow('Disabled reason', _disabledReason!),
                  _StatusRow('Latest alert', _latestAlertTime ?? 'None'),
                  _StatusRow('Stream', _streamActive ? 'Active' : 'Inactive'),
                  _StatusRow('Notifications', _notificationStatus),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _StatusRow extends StatelessWidget {
  const _StatusRow(this.label, this.value, {this.valueColor});
  final String label;
  final String value;
  final Color? valueColor;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(width: 120, child: Text(label)),
          Expanded(
            child: Text(
              value,
              style: TextStyle(fontWeight: FontWeight.w600, color: valueColor),
            ),
          ),
        ],
      ),
    );
  }
}

class LiveViewPage extends StatefulWidget {
  const LiveViewPage({required this.host, super.key});
  final String host;

  @override
  State<LiveViewPage> createState() => _LiveViewPageState();
}

class _LiveViewPageState extends State<LiveViewPage> {
  late final WebViewController _controller;
  String? _error;

  @override
  void initState() {
    super.initState();
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.disabled);
    _startStream();
  }

  Future<void> _startStream() async {
    try {
      await http
          .post(Uri.parse('${widget.host}/stream/start'))
          .timeout(const Duration(seconds: 5));
      if (mounted) {
        await _controller.loadRequest(Uri.parse('${widget.host}/video_feed'));
      }
    } catch (_) {
      if (mounted) setState(() => _error = 'Pi not reachable');
    }
  }

  Future<void> _stopStream() async {
    try {
      await http
          .post(Uri.parse('${widget.host}/stream/stop'))
          .timeout(const Duration(seconds: 2));
    } catch (_) {
      // Leaving the page should remain safe when the Pi is offline.
    }
  }

  @override
  void dispose() {
    unawaited(_stopStream());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Live View')),
      body: _error == null
          ? WebViewWidget(controller: _controller)
          : Center(child: Text(_error!)),
    );
  }
}

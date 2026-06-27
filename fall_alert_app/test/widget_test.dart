import 'package:fall_alert_app/main.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('shows fall monitor controls', (WidgetTester tester) async {
    await tester.pumpWidget(const FallAlertApp(notificationsReady: false));

    expect(find.text('Fall Detection Monitor'), findsOneWidget);
    expect(find.text('Connect'), findsOneWidget);
    expect(find.text('Check Status'), findsOneWidget);
    expect(find.text('Open Live View'), findsOneWidget);
  });
}

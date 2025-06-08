import 'package:app_flutter/features/home/home_screen.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:provider/provider.dart';
import 'package:app_flutter/core/services/notification_service.dart';
import 'package:app_flutter/providers/sms_provider.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await NotificationService.initialize();
  await Permission.notification.request();
  // Request permission at startup and store the result
  bool hasSmsPermission = await Permission.sms.status.isGranted;
  if (!hasSmsPermission) {
    final result = await Permission.sms.request();
    hasSmsPermission = result.isGranted;
    if (!hasSmsPermission && result.isPermanentlyDenied) {
      await openAppSettings();
    }
  }

  runApp(MyApp(hasSmsPermission: hasSmsPermission));
}

class MyApp extends StatelessWidget {
  final bool hasSmsPermission;

  const MyApp({required this.hasSmsPermission, Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => SMSProvider()),
      ],
      child: MaterialApp(
        title: 'SMS Spam Detection',
        debugShowCheckedModeBanner: false,
        theme: ThemeData(
          primarySwatch: Colors.blue,
        ),
        home: HomeScreen(),
      ),
    );
  }
}
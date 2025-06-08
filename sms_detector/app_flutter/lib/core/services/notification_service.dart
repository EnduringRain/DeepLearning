import 'package:app_flutter/shared/models/sms_model.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';
import 'package:flutter/material.dart';

class NotificationService {
  static final FlutterLocalNotificationsPlugin _notifications =
      FlutterLocalNotificationsPlugin();

  static Future<void> initialize() async {
    const AndroidInitializationSettings androidSettings =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    const InitializationSettings settings = InitializationSettings(
      android: androidSettings,
    );

    await _notifications.initialize(
      settings,
      onDidReceiveNotificationResponse: (NotificationResponse response) async {
        print('Notification tapped: ${response.payload}');
      },
    );

    const AndroidNotificationChannel spamChannel = AndroidNotificationChannel(
      'spam_alerts',
      'Spam Alerts',
      description: 'Notifications for detected spam messages',
      importance: Importance.high,
      playSound: true,
      enableVibration: true,
      showBadge: true,
    );

    const AndroidNotificationChannel normalChannel = AndroidNotificationChannel(
      'normal_messages',
      'Normal Messages',
      description: 'Notifications for normal messages',
      importance: Importance.defaultImportance,
      playSound: true,
      enableVibration: true,
      showBadge: true,
    );

    const AndroidNotificationChannel blockChannel = AndroidNotificationChannel(
      'block_notifications',
      'Block Notifications',
      description: 'Notifications for blocked messages',
      importance: Importance.defaultImportance,
      playSound: true,
      enableVibration: true,
      showBadge: true,
    );

    await _notifications
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(spamChannel);

    await _notifications
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(normalChannel);

    await _notifications
        .resolvePlatformSpecificImplementation<
            AndroidFlutterLocalNotificationsPlugin>()
        ?.createNotificationChannel(blockChannel);
  }

  static Future<void> showSpamAlert({
    required SMSModel sms,
    required double confidence,
  }) async {
    const AndroidNotificationDetails androidDetails = AndroidNotificationDetails(
      'spam_alerts',
      'Spam Alerts',
      channelDescription: 'Notifications for detected spam messages',
      importance: Importance.high,
      priority: Priority.high,
      color: Colors.red,
      icon: '@mipmap/ic_launcher',
    );

    const NotificationDetails details = NotificationDetails(
      android: androidDetails,
    );

    await _notifications.show(
      DateTime.now().millisecondsSinceEpoch.remainder(100000),
      'Spam Detected',
      'From: ${sms.address} (${(confidence * 100).toStringAsFixed(1)}% confidence)',
      details,
    );
  }

  static Future<void> showBlockNotification(String message) async {
    const AndroidNotificationDetails androidDetails = AndroidNotificationDetails(
      'block_notifications',
      'Block Notifications',
      channelDescription: 'Notifications for blocked messages',
      importance: Importance.defaultImportance,
      priority: Priority.defaultPriority,
      color: Colors.orange,
    );

    const NotificationDetails details = NotificationDetails(
      android: androidDetails,
    );

    await _notifications.show(
      DateTime.now().millisecondsSinceEpoch.remainder(100000),
      'Message Blocked',
      message,
      details,
    );
  }

  static Future<void> showNormalMessageNotification(SMSModel sms) async {
    const AndroidNotificationDetails androidDetails = AndroidNotificationDetails(
      'normal_messages',
      'Normal Messages',
      channelDescription: 'Notifications for normal messages',
      importance: Importance.defaultImportance,
      priority: Priority.defaultPriority,
      color: Colors.green,
      icon: '@mipmap/ic_launcher',
    );

    const NotificationDetails details = NotificationDetails(
      android: androidDetails,
    );

    await _notifications.show(
      DateTime.now().millisecondsSinceEpoch.remainder(100000),
      'Tin nhắn mới',
      'Từ: ${sms.address}',
      details,
    );
  }
}
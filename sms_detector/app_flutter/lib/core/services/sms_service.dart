import 'dart:async';
import 'package:sms_advanced/sms_advanced.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:app_flutter/shared/models/sms_model.dart';

class SMSService {
  static final SmsReceiver _receiver = SmsReceiver();
  static final SmsQuery _query = SmsQuery();
  static StreamSubscription<SmsMessage>? _smsSubscription;

  /// Kiểm tra quyền SMS mà không yêu cầu lại nếu đã có kết quả
  static Future<void> checkPermissions() async {
    final status = await Permission.sms.status;

    if (status.isPermanentlyDenied) {
      await openAppSettings();
    }
  }

  /// Lấy tất cả tin nhắn SMS từ hộp thư đến
  static Future<List<SMSModel>> getAllSMS({int limit = 100}) async {
    try {
      await checkPermissions();

      final messages = await _query.querySms(
        kinds: [SmsQueryKind.Inbox],
        count: limit,
      );

      print('Retrieved ${messages.length} SMS messages');

      return messages.map((msg) {
        return SMSModel(
          id: msg.id.toString(),
          address: msg.sender ?? 'Unknown',
          body: msg.body ?? '',
          date: DateTime.fromMillisecondsSinceEpoch(
            msg.date!.microsecondsSinceEpoch ?? 0,
          ),
          type: 1,
          prediction: null,
          confidence: null,
          isSpam: false,
        );
      }).toList();
    } catch (e) {
      print('Error reading SMS: $e');
      throw Exception('Không thể đọc tin nhắn: $e');
    }
  }

  /// Lấy các tin nhắn gần đây trong khoảng thời gian nhất định
  static Future<List<SMSModel>> getRecentSMS({
    int hours = 24,
    int limit = 100,
  }) async {
    try {
      await checkPermissions();

      final cutoffTime = DateTime.now().subtract(Duration(hours: hours));
      final messages = await _query.querySms(
        kinds: [SmsQueryKind.Inbox],
        count: limit,
        start: cutoffTime.millisecondsSinceEpoch,
      );

      return messages
          .where(
            (msg) => DateTime.fromMillisecondsSinceEpoch(
              msg.date!.microsecondsSinceEpoch ?? 0,
            ).isAfter(cutoffTime),
          )
          .map((msg) {
            return SMSModel(
              id: msg.id.toString(),
              address: msg.sender ?? 'Unknown',
              body: msg.body ?? '',
              date: DateTime.fromMillisecondsSinceEpoch(
                msg.date!.microsecondsSinceEpoch ?? 0,
              ),
              type: 1,
              prediction: null,
              confidence: null,
              isSpam: false,
            );
          })
          .toList();
    } catch (e) {
      print('Error reading recent SMS: $e');
      throw Exception('Không thể đọc tin nhắn gần đây: $e');
    }
  }

  /// Bắt đầu giám sát tin nhắn SMS theo thời gian thực
  static Future<void> startRealTimeMonitoring(
    Function(SMSModel) onNewMessageCallback,
  ) async {
    try {
      await checkPermissions();

      _smsSubscription?.cancel();
      if (_receiver.onSmsReceived != null) {
        _smsSubscription = _receiver.onSmsReceived!.listen((
          SmsMessage message,
        ) async {
          print('New SMS received: ${message.body} from ${message.sender}');
          final sms = SMSModel(
            id: message.id.toString(),
            address: message.sender ?? 'Unknown',
            body: message.body ?? '',
            date: DateTime.fromMillisecondsSinceEpoch(
              message.date!.microsecondsSinceEpoch ?? 0,
            ),
            type: 1,
            prediction: null,
            confidence: null,
            isSpam: false,
          );
  
          // Chỉ chuyển tin nhắn cho callback xử lý, không tự hiển thị thông báo
          onNewMessageCallback(sms);
        });
        print('Real-time SMS monitoring started using sms_advanced.');
      } else {
        print('SMS receiver stream is not available');
        throw Exception('SMS receiver stream is not available');
      }
    } catch (e) {
      print('Error starting real-time monitoring: $e');
      throw Exception('Failed to start real-time monitoring: $e');
    }
  }

  /// Dừng giám sát tin nhắn SMS theo thời gian thực
  static void stopRealTimeMonitoring() {
    _smsSubscription?.cancel();
    _smsSubscription = null;
    print('Real-time SMS monitoring stopped');
  }
}

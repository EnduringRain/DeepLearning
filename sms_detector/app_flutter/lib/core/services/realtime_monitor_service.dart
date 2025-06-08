import 'package:app_flutter/core/services/sms_service.dart';
import 'package:app_flutter/shared/models/sms_model.dart';

class RealtimeMonitorService {
  static bool _isMonitoring = false;

  static Future<void> startMonitoring(Function(SMSModel) onNewMessage) async {
    if (_isMonitoring) return;
    _isMonitoring = true;

    try {
      await SMSService.checkPermissions();
      await SMSService.startRealTimeMonitoring(onNewMessage);
      print('Real-time monitoring started');
    } catch (e) {
      _isMonitoring = false;
      print('Error starting real-time monitoring: $e');
      throw Exception('Failed to start real-time monitoring: $e');
    }
  }

  static void stopMonitoring() {
    if (!_isMonitoring) return;
    _isMonitoring = false;
    SMSService.stopRealTimeMonitoring();
    print('Real-time monitoring stopped');
  }
}
import 'package:app_flutter/core/services/realtime_monitor_service.dart';
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../shared/models/sms_model.dart';
import '../core/services/api_service.dart';
import '../core/services/sms_service.dart';
import '../core/services/notification_service.dart';

class SMSProvider extends ChangeNotifier {
  List<SMSModel> _messages = [];
  List<SMSModel> _spamMessages = [];
  List<SMSModel> _recentAnalyzed = [];
  bool _isLoading = false;
  bool _isRealTimeEnabled = false;
  double _detectionThreshold = 0.5;

  final ApiService _apiService = ApiService(baseUrl: 'http://10.0.2.2:8000');

  SMSProvider() {
    _loadRealTimeSettings();
  }

  Future<void> _loadRealTimeSettings() async {
    final prefs = await SharedPreferences.getInstance();
    final realTimeEnabled = prefs.getBool('real_time_analysis') ?? true;
    if (realTimeEnabled) {
      toggleRealTimeMonitoring(true);
    }
  }
  
  List<SMSModel> get messages => _messages;
  List<SMSModel> get spamMessages => _spamMessages;
  List<SMSModel> get recentAnalyzed => _recentAnalyzed;
  bool get isLoading => _isLoading;
  bool get isRealTimeEnabled => _isRealTimeEnabled;
  double get detectionThreshold => _detectionThreshold;

  int get totalMessages => _messages.length;
  int get spamCount => _spamMessages.length;
  int get hamCount => totalMessages - spamCount;
  double get spamPercentage =>
      totalMessages > 0 ? (spamCount / totalMessages) * 100 : 0;

  Future<void> loadMessages() async {
    _isLoading = true;
    notifyListeners();

    try {
      final messages = await SMSService.getAllSMS();
      _messages = messages;
      await batchAnalyzeMessages();
    } catch (e) {
      print('Error loading messages: $e');
    } finally {
      _isLoading = false;
      notifyListeners();
    }
  }

  Future<Map<String, dynamic>> analyzeMessage(String messageText) async {
    try {
      final result = await _apiService.predict(messageText);

      final sms = SMSModel(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        address: 'Manual Analysis',
        body: messageText,
        date: DateTime.now(),
        type: 1,
        prediction: result['prediction'],
        confidence: result['confidence'].toDouble(),
        isSpam: result['prediction'] == 'spam',
      );

      _recentAnalyzed.insert(0, sms);
      if (_recentAnalyzed.length > 50) {
        _recentAnalyzed = _recentAnalyzed.take(50).toList();
      }

      notifyListeners();
      return result;
    } catch (e) {
      throw Exception('Analysis failed: $e');
    }
  }

  Future<void> batchAnalyzeMessages({int? limit}) async {
  _isLoading = true;
  notifyListeners();

  try {
    final messagesToAnalyze =
        limit != null ? _messages.take(limit).toList() : _messages;
    final messageTexts = messagesToAnalyze.map((m) => m.body).toList();

    final results = await _apiService.batchPredict(messageTexts);

    for (int i = 0; i < messagesToAnalyze.length; i++) {
      final message = messagesToAnalyze[i];
      final result = results[i];

      final updatedMessage = message.copyWith(
        prediction: result['prediction'],
        confidence: result['confidence'].toDouble(),
        isSpam: result['prediction'] == 'spam',
      );

      final index = _messages.indexWhere((m) => m.id == message.id);
      if (index != -1) {
        _messages[index] = updatedMessage;
      }
    }

    _updateSpamMessages();
  } catch (e) {
    print('Batch analysis error: $e');
  } finally {
    _isLoading = false;
    notifyListeners();
  }
}

  void toggleRealTimeMonitoring(bool enabled) {
    _isRealTimeEnabled = enabled;
    if (enabled) {
      RealtimeMonitorService.startMonitoring(_handleNewSmsWithSpamCheck);
    } else {
      RealtimeMonitorService.stopMonitoring();
    }
    notifyListeners();
  }

  Future<void> _handleNewSmsWithSpamCheck(SMSModel sms) async {
  print('Handling new SMS: ${sms.body} from ${sms.address} at ${sms.date}');
  try {
    final result = await _apiService.predict(sms.body);
    print('API response for ${sms.body}: $result');
    final updatedSms = sms.copyWith(
      prediction: result['prediction'],
      confidence: result['confidence'].toDouble(),
      isSpam: result['prediction'] == 'spam',
    );

    _messages.insert(0, updatedSms);
    _recentAnalyzed.insert(0, updatedSms);
    if (_recentAnalyzed.length > 50) {
      _recentAnalyzed = _recentAnalyzed.take(50).toList();
    }

    _updateSpamMessages();

    if (updatedSms.isSpam == true) {
      await NotificationService.showSpamAlert(
        sms: updatedSms,
        confidence: updatedSms.confidence ?? 0.0,
      );
    } else {
      await NotificationService.showNormalMessageNotification(updatedSms);
    }
  } catch (e) {
    print('Error analyzing new SMS: $e');
    // Thêm tin nhắn vào danh sách nhưng đánh dấu là chưa phân tích
    final unanalyzedSms = sms.copyWith(
      prediction: 'unknown',
      confidence: 0.0,
      isSpam: false,
    );
    _messages.insert(0, unanalyzedSms);
    _recentAnalyzed.insert(0, unanalyzedSms);
    if (_recentAnalyzed.length > 50) {
      _recentAnalyzed = _recentAnalyzed.take(50).toList();
    }
    await NotificationService.showNormalMessageNotification(unanalyzedSms);
    // Thông báo lỗi cho người dùng nếu cần
    await NotificationService.showBlockNotification('Failed to analyze message: $e');
  }
  notifyListeners();
}

  Map<String, int> getSpamStatsByPeriod(int days) {
    final cutoffDate = DateTime.now().subtract(Duration(days: days));
    final recentSpam =
        _spamMessages.where((msg) => msg.date.isAfter(cutoffDate)).toList();

    Map<String, int> stats = {};
    for (var msg in recentSpam) {
      final dateKey = '${msg.date.day}/${msg.date.month}';
      stats[dateKey] = (stats[dateKey] ?? 0) + 1;
    }

    return stats;
  }

  Map<String, int> getTopSpamKeywords({int limit = 10}) {
    Map<String, int> keywords = {};

    for (var msg in _spamMessages) {
      final words = msg.body.toLowerCase().split(RegExp(r'\s+'));
      for (var word in words) {
        if (word.length > 3) {
          keywords[word] = (keywords[word] ?? 0) + 1;
        }
      }
    }

    var sortedEntries =
        keywords.entries.toList()..sort((a, b) => b.value.compareTo(a.value));

    return Map.fromEntries(sortedEntries.take(limit));
  }

  void _updateSpamMessages() {
    _spamMessages =
        _messages
            .where(
              (msg) =>
                  msg.isSpam == true &&
                  (msg.confidence ?? 0) > _detectionThreshold,
            )
            .toList();
  }

  void clearData() {
    _messages.clear();
    _spamMessages.clear();
    _recentAnalyzed.clear();
    notifyListeners();
  }

  Map<String, dynamic>? _modelInfo;
  Map<String, dynamic>? _feedbackStats;

  Map<String, dynamic>? get modelInfo => _modelInfo;
  Map<String, dynamic>? get feedbackStats => _feedbackStats;

  Future<void> fetchModelInfo() async {
    try {
      _modelInfo = await _apiService.getHealthStatus();
      notifyListeners();
    } catch (e) {
      print('Error fetching model info: $e');
    }
  }

  Future<void> fetchFeedbackStats() async {
    try {
      _feedbackStats = await _apiService.getFeedbackStats();
      notifyListeners();
    } catch (e) {
      print('Error fetching feedback stats: $e');
    }
  }

  Future<void> sendFeedback(
    String message,
    String prediction,
    bool isCorrect,
  ) async {
    try {
      await _apiService.submitFeedback(message, prediction, isCorrect);
      await fetchFeedbackStats();
    } catch (e) {
      print('Error sending feedback: $e');
      rethrow;
    }
  }
}

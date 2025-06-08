import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  final String baseUrl;

  ApiService({required this.baseUrl});

  Future<Map<String, dynamic>> predict(String message) async {
    try {
      final payload = {'text': message};
      print('Sending predict payload: ${jsonEncode(payload)}');
      final response = await http.post(
        Uri.parse('$baseUrl/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );

      print('Predict response: ${response.statusCode} - ${response.body}');
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to predict spam: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<List<Map<String, dynamic>>> batchPredict(List<String> messages) async {
  try {
    print('Sending batch predict request: ${jsonEncode({'messages': messages})}');
    final response = await http.post(
      Uri.parse('$baseUrl/batch_predict'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'messages': messages}),
    );

    print('Batch predict response: ${response.statusCode} - ${response.body}');
    if (response.statusCode == 200) {
      final List<dynamic> results = jsonDecode(response.body);
      return results.cast<Map<String, dynamic>>();
    } else {
      throw Exception('Failed to batch predict: ${response.statusCode} - ${response.body}');
    }
  } catch (e) {
    throw Exception('Network error: $e');
  }
}

  Future<Map<String, dynamic>> submitFeedback(
    String message,
    String prediction,
    bool isCorrect,
  ) async {
    try {
      final payload = {
        'message': message,
        'actual_label': isCorrect
            ? prediction.toLowerCase()
            : (prediction.toLowerCase() == 'spam' ? 'ham' : 'spam'),
        'predicted_label': prediction.toLowerCase(),
        'confidence': 0.0, // This will be updated later
      };
      print('Submitting feedback: ${jsonEncode(payload)}');
      final response = await http.post(
        Uri.parse('$baseUrl/feedback'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );

      print('Feedback response: ${response.statusCode} - ${response.body}');
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to submit feedback: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<Map<String, dynamic>> getHealthStatus() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/health'));

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to get health status: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<Map<String, dynamic>> getFeedbackStats() async {
    try {
      final response = await http.get(Uri.parse('$baseUrl/feedback_stats'));

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to get feedback stats: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }

  Future<Map<String, dynamic>> incrementalLearn({
    required List<Map<String, dynamic>> samples,
    required int epochs,
  }) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/incremental_learn'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'messages': samples,
          'epochs': epochs,
        }),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception('Failed to perform incremental learning: ${response.statusCode} - ${response.body}');
      }
    } catch (e) {
      throw Exception('Network error: $e');
    }
  }
}
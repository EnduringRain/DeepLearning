import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import '../../core/services/api_service.dart';

class ManualAnalysisScreen extends StatefulWidget {
  const ManualAnalysisScreen({super.key});

  @override
  _ManualAnalysisScreenState createState() => _ManualAnalysisScreenState();
}

class _ManualAnalysisScreenState extends State<ManualAnalysisScreen> {
  final TextEditingController _messageController = TextEditingController();
  bool _isAnalyzing = false;
  String? _result;
  double? _confidence;

  @override
  void dispose() {
    _messageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Phân Tích Tin Nhắn'),
        backgroundColor: const Color(0xFF1565C0),
        foregroundColor: Colors.white,
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.grey[50]!, Colors.white],
          ),
        ),
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildInputCard(),
              const SizedBox(height: 20),
              _buildAnalyzeButton(),
              const SizedBox(height: 20),
              if (_result != null) _buildResultCard(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildInputCard() {
    return Card(
      elevation: 8,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          gradient: LinearGradient(colors: [Colors.white, Colors.grey[50]!]),
        ),
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Nội dung tin nhắn',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.bold,
                color: Color(0xFF1565C0),
              ),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _messageController,
              maxLines: 6,
              decoration: InputDecoration(
                hintText: 'Dán nội dung tin nhắn cần kiểm tra...',
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: BorderSide(color: Colors.grey[300]!),
                ),
                focusedBorder: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                  borderSide: const BorderSide(color: Color(0xFF1565C0), width: 2),
                ),
                filled: true,
                fillColor: Colors.grey[50],
              ),
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                TextButton.icon(
                  onPressed: _pasteFromClipboard,
                  icon: const Icon(Icons.paste, size: 18),
                  label: const Text('Dán từ clipboard'),
                  style: TextButton.styleFrom(
                    foregroundColor: const Color(0xFF1565C0),
                  ),
                ),
                const Spacer(),
                TextButton.icon(
                  onPressed: _clearText,
                  icon: const Icon(Icons.clear, size: 18),
                  label: const Text('Xóa'),
                  style: TextButton.styleFrom(
                    foregroundColor: Colors.grey[600],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAnalyzeButton() {
    return Container(
      height: 56,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        gradient: const LinearGradient(
          colors: [Color(0xFF1565C0), Color(0xFF1976D2)],
        ),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF1565C0).withOpacity(0.3),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ElevatedButton(
        onPressed: _isAnalyzing ? null : _analyzeMessage,
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.transparent,
          shadowColor: Colors.transparent,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
        ),
        child: _isAnalyzing
            ? Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: const [
                  SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                    ),
                  ),
                  SizedBox(width: 12),
                  Text(
                    'Đang phân tích...',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ],
              )
            : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: const [
                  Icon(Icons.security, color: Colors.white),
                  SizedBox(width: 8),
                  Text(
                    'Kiểm Tra Spam',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ],
              ),
      ),
    );
  }

  Widget _buildResultCard() {
    final isSpam = _result == 'spam';
    final color = isSpam ? const Color(0xFFF44336) : const Color(0xFF4CAF50);
    final icon = isSpam ? Icons.warning : Icons.check_circle;
    final title = isSpam ? 'SPAM PHÁT HIỆN' : 'TIN NHẮN AN TOÀN';

    return Card(
      elevation: 12,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Container(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          gradient: LinearGradient(
            colors: [color.withOpacity(0.1), Colors.white],
          ),
          border: Border.all(color: color.withOpacity(0.3), width: 2),
        ),
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Icon(icon, size: 48, color: color),
            const SizedBox(height: 12),
            Text(
              title,
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: color,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Độ tin cậy: ${(_confidence! * 100).toStringAsFixed(1)}%',
              style: TextStyle(fontSize: 14, color: Colors.grey[600]),
            ),
            const SizedBox(height: 16),
            _buildFeedbackButtons(),
          ],
        ),
      ),
    );
  }

  Widget _buildFeedbackButtons() {
    return Row(
      children: [
        Expanded(
          child: OutlinedButton.icon(
            onPressed: () => _provideFeedback(true),
            icon: const Icon(Icons.thumb_up, size: 18),
            label: const Text('Chính xác'),
            style: OutlinedButton.styleFrom(
              foregroundColor: const Color(0xFF4CAF50),
              side: const BorderSide(color: Color(0xFF4CAF50)),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: OutlinedButton.icon(
            onPressed: () => _provideFeedback(false),
            icon: const Icon(Icons.thumb_down, size: 18),
            label: const Text('Sai'),
            style: OutlinedButton.styleFrom(
              foregroundColor: const Color(0xFFF44336),
              side: const BorderSide(color: Color(0xFFF44336)),
            ),
          ),
        ),
      ],
    );
  }

  Future<void> _pasteFromClipboard() async {
    final clipboardData = await Clipboard.getData('text/plain');
    if (clipboardData?.text != null) {
      setState(() {
        _messageController.text = clipboardData!.text!;
      });
    }
  }

  void _clearText() {
    setState(() {
      _messageController.clear();
      _result = null;
      _confidence = null;
    });
  }

  Future<void> _analyzeMessage() async {
    final message = _messageController.text.trim();
    if (message.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Vui lòng nhập nội dung tin nhắn để phân tích'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }
    if (message.length < 10) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Tin nhắn phải có ít nhất 10 ký tự'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    setState(() {
      _isAnalyzing = true;
    });

    try {
      final apiService = ApiService(baseUrl: 'http://10.0.2.2:8000');
      final response = await apiService.predict(message);

      setState(() {
        _result = response['prediction'] as String;
        _confidence = (response['confidence'] as num).toDouble();
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Lỗi khi phân tích: $e'),
          backgroundColor: Colors.red,
        ),
      );
    } finally {
      setState(() {
        _isAnalyzing = false;
      });
    }
  }

  Future<void> _provideFeedback(bool isCorrect) async {
    if (_result == null || _confidence == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Không có kết quả để gửi phản hồi'),
          backgroundColor: Colors.red,
        ),
      );
      return;
    }

    try {
      final apiService = ApiService(baseUrl: 'http://10.0.2.2:8000');
      await apiService.submitFeedback(
        _messageController.text,
        _result!,
        isCorrect,
      );

      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            isCorrect
                ? 'Cảm ơn phản hồi của bạn!'
                : 'Chúng tôi sẽ cải thiện độ chính xác',
          ),
          backgroundColor: isCorrect ? Colors.green : Colors.orange,
        ),
      );
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Lỗi khi gửi phản hồi: $e'),
          backgroundColor: Colors.red,
        ),
      );
    }
  }
}
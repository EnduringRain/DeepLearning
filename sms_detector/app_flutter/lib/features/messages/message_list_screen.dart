import 'package:app_flutter/shared/models/sms_model.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../providers/sms_provider.dart';

class MessageListScreen extends StatelessWidget {
  const MessageListScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Danh sách tin nhắn'),
        backgroundColor: const Color(0xFF1565C0),
        foregroundColor: Colors.white,
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: () {
              context.read<SMSProvider>().loadMessages();
            },
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.grey[50]!, Colors.white],
          ),
        ),
        child: Consumer<SMSProvider>(
          builder: (context, provider, child) {
            if (provider.isLoading) {
              return const Center(child: CircularProgressIndicator());
            }
            
            if (provider.messages.isEmpty) {
              return Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.message, size: 64, color: Colors.grey[400]),
                    const SizedBox(height: 16),
                    Text(
                      'Không có tin nhắn nào',
                      style: TextStyle(
                        fontSize: 18,
                        color: Colors.grey[600],
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              );
            }
            
            return ListView.builder(
              itemCount: provider.messages.length,
              itemBuilder: (context, index) {
                final message = provider.messages[index];
                final isSpam = message.isSpam == true;
                
                return Card(
                  elevation: 4,
                  margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                    side: BorderSide(
                      color: isSpam ? Colors.red.withOpacity(0.3) : Colors.transparent,
                      width: isSpam ? 1 : 0,
                    ),
                  ),
                  child: ExpansionTile(
                    leading: CircleAvatar(
                      backgroundColor: isSpam ? Colors.red[100] : Colors.blue[100],
                      child: Icon(
                        isSpam ? Icons.warning : Icons.message,
                        color: isSpam ? Colors.red : const Color(0xFF1565C0),
                      ),
                    ),
                    title: Text(
                      message.address,
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: isSpam ? Colors.red : Colors.black,
                      ),
                    ),
                    subtitle: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          message.shortBody,
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            Text(
                              message.displayDate,
                              style: TextStyle(fontSize: 12, color: Colors.grey[600]),
                            ),
                            if (isSpam)
                              Container(
                                margin: const EdgeInsets.only(left: 8),
                                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                decoration: BoxDecoration(
                                  color: Colors.red,
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: Text(
                                  'SPAM',
                                  style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 10,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                          ],
                        ),
                      ],
                    ),
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(16.0),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              message.body,
                              style: const TextStyle(fontSize: 14),
                            ),
                            const SizedBox(height: 16),
                            if (isSpam)
                              Text(
                                'Độ tin cậy: ${((message.confidence ?? 0) * 100).toStringAsFixed(1)}%',
                                style: TextStyle(
                                  color: Colors.grey[600],
                                  fontStyle: FontStyle.italic,
                                ),
                              ),
                            const SizedBox(height: 16),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.end,
                              children: [
                                OutlinedButton.icon(
                                  onPressed: () => _provideFeedback(context, message, true),
                                  icon: const Icon(Icons.thumb_up, size: 16),
                                  label: const Text('Chính xác'),
                                  style: OutlinedButton.styleFrom(
                                    foregroundColor: Colors.green,
                                    side: const BorderSide(color: Colors.green),
                                  ),
                                ),
                                const SizedBox(width: 8),
                                OutlinedButton.icon(
                                  onPressed: () => _provideFeedback(context, message, false),
                                  icon: const Icon(Icons.thumb_down, size: 16),
                                  label: const Text('Sai'),
                                  style: OutlinedButton.styleFrom(
                                    foregroundColor: Colors.red,
                                    side: const BorderSide(color: Colors.red),
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                );
              },
            );
          },
        ),
      ),
    );
  }

  void _provideFeedback(BuildContext context, SMSModel message, bool isCorrect) async {
    try {
      await context.read<SMSProvider>().sendFeedback(
        message.body,
        message.prediction ?? 'unknown',
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
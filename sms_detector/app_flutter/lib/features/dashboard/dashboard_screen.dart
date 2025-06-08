import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../../providers/sms_provider.dart';
import 'package:fl_chart/fl_chart.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({Key? key}) : super(key: key);

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {

  @override
  void initState() {
    super.initState();
    _loadData();
  }

  Future<void> _loadData() async {
    final provider = Provider.of<SMSProvider>(context, listen: false);
    await provider.fetchModelInfo();
    await provider.fetchFeedbackStats();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Thông Số Mô Hình'),
        backgroundColor: const Color(0xFF1565C0),
        foregroundColor: Colors.white,
      ),
      body: Consumer<SMSProvider>(
        builder: (context, provider, child) {
          if (provider.isLoading) {
            return const Center(child: CircularProgressIndicator());
          }

          return RefreshIndicator(
            onRefresh: _loadData,
            child: SingleChildScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildModelInfoCard(provider),
                  const SizedBox(height: 20),
                  _buildSpamHamPieChart(provider),
                  const SizedBox(height: 20),
                  _buildTopKeywords(provider),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildModelInfoCard(SMSProvider provider) {
  final modelInfo = provider.modelInfo;

  return Card(
    elevation: 4,
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
    child: Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Thông Tin Mô Hình',
            style: TextStyle(
              fontSize: 18,
              fontWeight: FontWeight.bold,
              color: Color(0xFF1565C0),
            ),
          ),
          const SizedBox(height: 12),
          if (modelInfo == null)
            const Text('Không có thông tin mô hình')
          else
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildInfoRow(
                  'Trạng thái',
                  (modelInfo['model_loaded'] as bool? ?? false) ? 'Đã tải' : 'Chưa tải',
                ),
                _buildInfoRow(
                  'Kích thước từ điển',
                  '${modelInfo['word_index_size'] ?? 0} từ',
                ),
                _buildInfoRow(
                  'Ngưỡng phát hiện',
                  '${((modelInfo['threshold'] as double? ?? 0.5) * 100).toStringAsFixed(1)}%',
                ),
                _buildInfoRow(
                  'Số lượng feedback',
                  '${modelInfo['feedback_count'] ?? 0}',
                ),
                _buildInfoRow(
                  'Đang huấn luyện',
                  (modelInfo['retrain_status'] == 'running') ? 'Có' : 'Không',
                ),
              ],
            ),
        ],
      ),
    ),
  );
}


  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: TextStyle(fontSize: 14, color: Colors.grey[700])),
          Text(
            value,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildSpamHamPieChart(SMSProvider provider) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Phân Bố Tin Nhắn',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Color(0xFF1565C0),
              ),
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200,
              child: PieChart(
                PieChartData(
                  sections: [
                    PieChartSectionData(
                      value: provider.spamCount.toDouble(),
                      title: 'Spam${provider.spamCount}',
                      color: Colors.red,
                      radius: 80,
                    ),
                    PieChartSectionData(
                      value: provider.hamCount.toDouble(),
                      title: 'An toàn${provider.hamCount}',
                      color: Colors.green,
                      radius: 80,
                    ),
                  ],
                  sectionsSpace: 2,
                  centerSpaceRadius: 40,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTopKeywords(SMSProvider provider) {
    final keywords = provider.getTopSpamKeywords(limit: 5);

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Từ Khóa Spam Phổ Biến',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
                color: Color(0xFF1565C0),
              ),
            ),
            const SizedBox(height: 12),
            if (keywords.isEmpty)
              const Text('Không có từ khóa spam')
            else
              Column(
                children:
                    keywords.entries.map((entry) {
                      return Padding(
                        padding: const EdgeInsets.only(bottom: 8),
                        child: Row(
                          children: [
                            Expanded(flex: 3, child: Text(entry.key)),
                            Expanded(
                              flex: 7,
                              child: LinearProgressIndicator(
                                value:
                                    entry.value / keywords.entries.first.value,
                                backgroundColor: Colors.grey[200],
                                valueColor: const AlwaysStoppedAnimation<Color>(
                                  Colors.red,
                                ),
                              ),
                            ),
                            const SizedBox(width: 8),
                            Text(
                              '${entry.value}',
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      );
                    }).toList(),
              ),
          ],
        ),
      ),
    );
  }
}

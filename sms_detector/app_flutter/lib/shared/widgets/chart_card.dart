import 'package:flutter/material.dart';

class ChartCard extends StatelessWidget {
  final String title;
  final Widget child;
  final List<Widget>? actions;

  const ChartCard({
    Key? key,
    required this.title,
    required this.child,
    this.actions,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min, // Đảm bảo Column không chiếm quá nhiều không gian
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                if (actions != null) ...actions!,
              ],
            ),
            const SizedBox(height: 16),
            SizedBox(
              height: 200, // Chiều cao cố định cho nội dung
              width: double.infinity,
              child: child,
            ),
          ],
        ),
      ),
    );
  }
}
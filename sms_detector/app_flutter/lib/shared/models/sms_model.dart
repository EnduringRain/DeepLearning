class SMSModel {
  final String id;
  final String address;
  final String body;
  final DateTime date;
  final int type; // 1: inbox, 2: sent
  String? prediction;
  double? confidence;
  bool? isSpam;
  bool? userFeedback; // true if user confirmed, false if user corrected
  
  SMSModel({
    required this.id,
    required this.address,
    required this.body,
    required this.date,
    required this.type,
    this.prediction,
    this.confidence,
    this.isSpam,
    this.userFeedback,
  });
  
  factory SMSModel.fromMap(Map<String, dynamic> map) {
    return SMSModel(
      id: map['id']?.toString() ?? '',
      address: map['address']?.toString() ?? '',
      body: map['body']?.toString() ?? '',
      date: DateTime.fromMillisecondsSinceEpoch(
        int.tryParse(map['date']?.toString() ?? '0') ?? 0,
      ),
      type: int.tryParse(map['type']?.toString() ?? '1') ?? 1,
      prediction: map['prediction']?.toString(),
      confidence: double.tryParse(map['confidence']?.toString() ?? '0'),
      isSpam: map['isSpam'] as bool?,
      userFeedback: map['userFeedback'] as bool?,
    );
  }
  
  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'address': address,
      'body': body,
      'date': date.millisecondsSinceEpoch.toString(),
      'type': type.toString(),
      'prediction': prediction,
      'confidence': confidence,
      'isSpam': isSpam,
      'userFeedback': userFeedback,
    };
  }
  
  SMSModel copyWith({
    String? prediction,
    double? confidence,
    bool? isSpam,
    bool? userFeedback,
  }) {
    return SMSModel(
      id: id,
      address: address,
      body: body,
      date: date,
      type: type,
      prediction: prediction ?? this.prediction,
      confidence: confidence ?? this.confidence,
      isSpam: isSpam ?? this.isSpam,
      userFeedback: userFeedback ?? this.userFeedback,
    );
  }
  
  String get displayDate {
    final now = DateTime.now();
    final difference = now.difference(date);
    
    if (difference.inDays > 0) {
      return '${difference.inDays} ngày trước';
    } else if (difference.inHours > 0) {
      return '${difference.inHours} giờ trước';
    } else if (difference.inMinutes > 0) {
      return '${difference.inMinutes} phút trước';
    } else {
      return 'Vừa xong';
    }
  }
  
  String get shortBody {
    return body.length > 50 ? '${body.substring(0, 50)}...' : body;
  }
}
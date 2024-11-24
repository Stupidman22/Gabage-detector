#include <Servo.h>

// Định nghĩa các chân servo và phần cứng khác
#define SERVO1_PIN 3
#define SERVO2_PIN 5
#define SERVO3_PIN 9
#define SERVO4_PIN 10
#define SENSOR_PIN 12
#define RELAY_PIN 7

// Khai báo đối tượng servo
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;

void setup() {
  pinMode(SENSOR_PIN, INPUT);     // Cấu hình cảm biến làm đầu vào
  pinMode(RELAY_PIN, OUTPUT);    // Cấu hình relay làm đầu ra
  digitalWrite(RELAY_PIN, HIGH); // Mặc định relay bật (HIGH)

  // Gắn các chân servo với các đối tượng Servo
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo3.attach(SERVO3_PIN);
  servo4.attach(SERVO4_PIN);

  // Đặt tất cả các servo về vị trí ban đầu (0 độ)
  servo1.write(0);
  servo2.write(0);
  servo3.write(0);
  servo4.write(0);

  // Thiết lập UART
  Serial.begin(9600); // Tốc độ baud
}

void loop() {
  static bool waitingForPi = false; // Cờ để xác định trạng thái chờ phản hồi từ Pi 5
  static unsigned long sendTime = 0; // Thời gian gửi tín hiệu `0`
  // digitalWrite(RELAY_PIN, LOW);  // Tat di
  // Kiểm tra trạng thái cảm biến
  if (digitalRead(SENSOR_PIN) == LOW && !waitingForPi) {
    digitalWrite(RELAY_PIN, LOW);  // Tắt relay
    Serial.println(0);            // Gửi giá trị `0` lên Pi 5
    sendTime = millis();          // Lưu thời gian gửi
    waitingForPi = true;          // Bật cờ chờ phản hồi
  }

  // Nếu đang chờ phản hồi từ Pi 5
  if (waitingForPi) {
    if (Serial.available() > 0) {
      char input = Serial.read(); // Đọc giá trị từ Pi 5
      // Reset trạng thái
      digitalWrite(RELAY_PIN, HIGH); // Bật lại relay
      // Xử lý giá trị nhận được
      switch (input) {
        case '1':
          activateServo(servo1, 7000);
          break;
        case '2':
          activateServo(servo2, 12000);
          break;
        case '3':
          activateServo(servo3, 17000);
          break;
        case '4':
          activateServo(servo4, 22000);
          break;
        case '5':
          // Không gạt servo nào
          // No action, but explicitly reset waitingForPi
          Serial.println("Plastic detected, no servo action.");
          break;
        default:
          Serial.println("Invalid input"); // Giá trị không hợp lệ
          break;
      }
      waitingForPi = false;          // Thoát trạng thái chờ
    }

    // Nếu không nhận được phản hồi trong 5 giây
    if (millis() - sendTime >= 5000) {
      digitalWrite(RELAY_PIN, HIGH); // Bật lại relay
      delay(5000);
      waitingForPi = false;          // Thoát trạng thái chờ
    }
  }
}

// Hàm gạt servo
void activateServo(Servo& servo, unsigned long delayTime) {
  servo.write(50);        // Gạt servo sang góc 45 độ
  delay(delayTime);       // Giữ servo trong khoảng thời gian chỉ định
  servo.write(0);         // Đưa servo trở lại vị trí ban đầu
}

#include <Servo.h>

Servo esc;
const int escPin = 3;
Servo motor;
int motorPin = 2;
const int buttonPin = 4;

// --- Global Sensor Arrays ---
const int trigPins[] = {6, 5, 7, 8, 9, 10, 11, 12, 13, 14}; 
const int echoPins[] = {23, 22, 24, 25, 26, 27, 28, 29, 30, 31};
const String labels[] = {
  "front", "front_left", "front_right", "side_right_front", "side_right_back", 
  "side_left_front", "side_left_back", "rear", "rear_left", "rear_right"
};

void setup() {
  Serial.begin(115200);
  pinMode(buttonPin, INPUT_PULLUP);

  motor.attach(motorPin);
  delay(2000);
  esc.attach(escPin, 1000, 2000);
  delay(1000);

  for (int i = 0; i < 10; i++) { // Corrected to 10 sensors
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT_PULLUP);
    // For MB1010 in PW mode, pulling Trig high or leaving it low 
    // changes ranging behavior. Usually, we leave it LOW to trigger manually.
    digitalWrite(trigPins[i], LOW);
  }
}

void loop() {
  sendSensorData();

  if (getDistanceMeters(0) <= 0.5) { 
    emergencyStop();           // Call your controlled stop function
    delay(5000);      // Wait for 5 seconds
  }

  if (digitalRead(buttonPin) == LOW) {
    emergencyStop();

    exit(0);
  }

  if (Serial.available() > 0) {
    char cmd = Serial.read();
    handleCommand(cmd);
  }
}

// Function to read sensors (Handles both HC-SR04 and MB1010)
float getDistanceMeters(int index) {
  int trig = trigPins[index];
  int echo = echoPins[index];

  // 1. Ensure a clean start
  digitalWrite(trig, LOW);
  delayMicroseconds(5);

  // 2. Trigger pulse (MB1010 needs 20us+)
  digitalWrite(trig, HIGH);
  delayMicroseconds(25); 
  digitalWrite(trig, LOW);

  // 3. Measure pulse width (Timeout of 40ms)
  long duration = pulseIn(echo, HIGH, 40000); 
  if (duration == 0) return 9.99;

  // LOGIC SPLIT:
  if (index == 0 || index == 1 || index == 2 || index >= 7) {
    // MB1010 ADJUSTED: Based on your 11m reading for a 2m ceiling, 
    // your sensor is likely using 58uS per cm (Standard for many MaxSonars).
    // Formula: (Duration / 58.0) / 100.0 to get meters
    float meters = (duration / 5800.0);
    return meters;
  } else {
    // HC-SR04: Standard Math
    return (duration * 0.034 / 2.0) / 100.0;
  }
}

void sendSensorData() {
  static unsigned long lastUpdate = 0;
  if (millis() - lastUpdate > 100) { 
    Serial.print("{");
    
    for (int i = 0; i < 10; i++) {
      float distanceMeters = getDistanceMeters(i);
      delay(30);
      
      Serial.print("\"");
      Serial.print(labels[i]);
      Serial.print("\": ");
      Serial.print(distanceMeters, 2);
      
      if (i < 9) Serial.print(", ");
    }
    
    Serial.println("}"); 
    lastUpdate = millis();
  }
}

// --- Rest of your movement logic ---
void handleCommand(char cmd) {
  switch(cmd) {
    case 'd': drive(); break;
    case 's': stop(); break;
    case 'e': emergencyStop(); break;
    case 'r': right(); break;
    case 'l': left(); break;
    case 'f': straight(); break;
    case 'v': reverse(); break;
    case 'o': rstop(); break;
    case 'b': backStraight(); break;
  }
}

void smartWait(unsigned long ms) {
  unsigned long start = millis();
  while (millis() - start < ms) {
    if (digitalRead(buttonPin) == LOW) emergencyStop();
    if (Serial.available() > 0 && Serial.peek() == 'e') {
        Serial.read(); // clear the 'e'
        emergencyStop();
    }
    sendSensorData(); 
  }
}

void drive() { esc.writeMicroseconds(1400); smartWait(1000); esc.writeMicroseconds(1350); smartWait(1000); esc.writeMicroseconds(1300); }
void stop() { esc.writeMicroseconds(1350); smartWait(1000); esc.writeMicroseconds(1400); smartWait(1000); esc.writeMicroseconds(1500); }
void left() { motor.writeMicroseconds(2000); }
void right() { motor.writeMicroseconds(1000); }
void straight() { motor.writeMicroseconds(1480); }
void emergencyStop() { esc.writeMicroseconds(1500); smartWait(5000); }
void reverse() { esc.writeMicroseconds(1600); smartWait(1000); esc.writeMicroseconds(1650); smartWait(1000); esc.writeMicroseconds(1700); }
void rstop() { esc.writeMicroseconds(1650); smartWait(1000); esc.writeMicroseconds(1600); smartWait(1000); esc.writeMicroseconds(1500); }
void backStraight() { motor.writeMicroseconds(1510); }

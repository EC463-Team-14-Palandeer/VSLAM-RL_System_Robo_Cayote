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

  for (int i = 0; i < 10; i++) { 
    pinMode(trigPins[i], OUTPUT);
    pinMode(echoPins[i], INPUT_PULLUP);
    digitalWrite(trigPins[i], LOW);
  }
}

void loop() {
  // 1. Report Environment to Jetson
  sendSensorData();

  // 2. Hardware-level Safety Override (Instant, Non-blocking)
  if (getDistanceMeters(0) <= 0.45) { 
    emergencyStop(); 
  }

  // 3. Physical Kill Switch
  if (digitalRead(buttonPin) == LOW) {
    emergencyStop();
  }

  // 4. Listen for Jetson Commands
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    handleCommand(cmd);
  }
}

// Function to read sensors (Unchanged, works perfectly)
float getDistanceMeters(int index) {
  int trig = trigPins[index];
  int echo = echoPins[index];

  digitalWrite(trig, LOW);
  delayMicroseconds(5);

  digitalWrite(trig, HIGH);
  delayMicroseconds(25); 
  digitalWrite(trig, LOW);

  long duration = pulseIn(echo, HIGH, 40000); 
  if (duration == 0) return 9.99;

  if (index == 0 || index == 1 || index == 2 || index >= 7) {
    float meters = (duration / 5800.0);
    return meters;
  } else {
    return (duration * 0.034 / 2.0) / 100.0;
  }
}

// Function to send JSON to Jetson
void sendSensorData() {
  static unsigned long lastUpdate = 0;
  if (millis() - lastUpdate > 100) { 
    Serial.print("{");
    
    for (int i = 0; i < 10; i++) {
      float distanceMeters = getDistanceMeters(i);
      delay(20); // Slightly reduced delay to speed up 10-sensor sweep
      
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

// --- INSTANT MOVEMENT LOGIC ---
// No more delays! The Jetson handles how long these run.
void handleCommand(char cmd) {
  switch(cmd) {
    case 'd': drive(); break;
    case 's': stop(); break;
    case 'e': emergencyStop(); break;
    case 'r': right(); break;
    case 'l': left(); break;
    case 'f': straight(); break;
    case 'v': reverse(); break;
    case 'o': stop(); break; // Combined with standard stop
    case 'b': backStraight(); break;
  }
}

// Assuming 1500 is neutral/stopped for ESC and ~1480 is straight for Steering
void drive() { esc.writeMicroseconds(1400); }
void stop() { esc.writeMicroseconds(1500); }
void left() { motor.writeMicroseconds(2000); }
void right() { motor.writeMicroseconds(1000); }
void straight() { motor.writeMicroseconds(1480); }
void reverse() { esc.writeMicroseconds(1600); }
void backStraight() { motor.writeMicroseconds(1510); }

void emergencyStop() { 
  esc.writeMicroseconds(1500); 
  motor.writeMicroseconds(1480); 
}

#include <ESP32Servo.h>
#include <Stepper.h>
#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>


// Shooter servo positions for ARM game
int pan_arm_position = 75;   
int tilt_arm_position = 180;  

// Shooter servo positions for RUBIK game
int pan_rubik_position = 160; 
int tilt_rubik_position = 180;

// Relays
const int armRelayPin = 32;       // Relay for arm
const int magnetRelayPin = 33;    // Relay for magnet (used in arm game)
// const int shooterRelayPin = 19;   // Relay for shooter
// const int rubikRelayPin = 6;     // Relay for Rubik

// Arm Game Variables
Servo arm_servo1, arm_servo2, arm_servo3;
const int arm_servo1_pin = 26;
const int arm_servo2_pin = 27;
const int arm_servo3_pin = 14;
int magnet_state;
int arm_pickSig;
int arm_flg;

// Shooter Game Variables
Servo shooter_servo1; // MG996R
Servo shooter_servo2; // S3003
const int shooter_servo1_pin = 13;
const int shooter_servo2_pin = 12;
// const int shooter_step_pin = 23;
// const int shooter_dir_pin = 21;
// const int shooter_switch_pin = 18;
// const int shooter_enable_pin = 7;
int total_steps = 0;
int shooter_flag_home = 0;
int shooter_reverse_direction = 0;
const int shooter_steps_per_rev = 200;
int shooter_motor_speed = 1000;
const int minAngle1 = 0;
const int maxAngle1 = 180;
const int minAngle2 = 0;
const int maxAngle2 = 270;


// Rubik's Cube Game Variables
const int robic_dir_pin = 15;
const int robic_step_pin = 2;
const int robic_enable_pin_front = 0;
const int robic_enable_pin_left = 4;
const int robic_enable_pin_right = 5;
const int robic_enable_pin_back = 22;
const int robic_enable_pin_down = 19;
const int STEPS_PER_REV = 6400;
const int MOTOR_RPM = 25;
const int STEPS_PER_90 = 200;
Stepper robic_stepper(STEPS_PER_REV, robic_step_pin, robic_dir_pin);
const int enablePins[5] = {robic_enable_pin_back, robic_enable_pin_right, robic_enable_pin_left, robic_enable_pin_down, robic_enable_pin_front};
enum Motor { BACK, RIGHT, LEFT, DOWN, FRONT };

String currentGame = "";
WebSocketsServer webSocket = WebSocketsServer(80);

// Function declarations
void setup_arm();
void setup_robic();
void setup_shooter();
bool play_arm(String command);
bool play_robic(String command);
bool play_shooter(String command);
void smoothMoveSync(Servo& s1, int& curr1, int target1, Servo& s2, int& curr2, int target2, Servo& s3, int& curr3, int target3, int delayTime, int pickSig);
void executeSolution(String solution);
void enableMotor(Motor motor);
void moveStepper(Motor motor, bool clockwise, int steps = 1);
void shoot();
void rotateSteps(int steps);
void setGameRelays();
void resetCurrentGame();
void webSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length);

void setGameRelays() {
  digitalWrite(armRelayPin, currentGame == "ARM" ? LOW : HIGH);
  // digitalWrite(shooter_enable_pin, currentGame == "SHOOTER" ? LOW : HIGH);
  // digitalWrite(rubikRelayPin, currentGame == "RUBIK" ? LOW : HIGH);
  digitalWrite(magnetRelayPin, HIGH);  
}

void setup() {
  Serial.begin(9600);
  pinMode(armRelayPin, OUTPUT);
  pinMode(magnetRelayPin, OUTPUT);
  // pinMode(shooter_enable_pin, OUTPUT);
  // pinMode(rubikRelayPin, OUTPUT);
  digitalWrite(armRelayPin, HIGH);
  digitalWrite(magnetRelayPin, HIGH);
  // digitalWrite(shooter_enable_pin, HIGH);
  // digitalWrite(rubikRelayPin, HIGH);

  setup_arm();
  setup_robic();
  setup_shooter();

  const char* ssid = "LEH";
  const char* password = "Leh@#1234";
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");
  Serial.println(WiFi.localIP());

  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  Serial.println("WebSocket server started");
}

void resetCurrentGame() {
  if (currentGame == "ARM") play_arm("180,0,0,0,0");
  else if (currentGame == "SHOOTER") play_shooter("90 90");
  else if (currentGame == "RUBIK") play_robic(" ");
}

void loop() {
  webSocket.loop();
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.printf("Client %u disconnected\n", num);
      break;
    case WStype_CONNECTED:
      Serial.printf("Client %u connected\n", num);
      break;
    case WStype_TEXT:
      DynamicJsonDocument doc(1024);
      DeserializationError error = deserializeJson(doc, payload);
      if (error) {
        webSocket.sendTXT(num, "{\"success\": false, \"message\": \"Invalid JSON\"}");
        return;
      }
      if (!doc.containsKey("action")) {
        webSocket.sendTXT(num, "{\"success\": false, \"message\": \"Missing 'action' in request\"}");
        return;
      }
      String action = doc["action"];
      if (action == "switch") {
        if (!doc.containsKey("game")) {
          webSocket.sendTXT(num, "{\"success\": false, \"message\": \"Missing 'game' in request\"}");
          return;
        }
        String newGame = doc["game"];
        newGame.toUpperCase();
        if (newGame == "ARM" || newGame == "RUBIK" || newGame == "SHOOTER") {
          if (currentGame != newGame) {
            if (currentGame != "") resetCurrentGame();
            currentGame = newGame;
            setGameRelays();
            if (currentGame == "ARM") {
              shooter_servo1.write(pan_arm_position);
              shooter_servo2.write(tilt_arm_position);
            } else if (currentGame == "RUBIK") {
              shooter_servo1.write(pan_rubik_position);
              shooter_servo2.write(tilt_rubik_position);
            }
          }
          String response = "{\"success\": true, \"message\": \"Switched to " + currentGame + " game\"}";
          webSocket.sendTXT(num, response);
        } else {
          String response = "{\"success\": false, \"message\": \"Invalid game: " + newGame + "\"}";
          webSocket.sendTXT(num, response);
        }
      } else if (action == "command") {
        if (currentGame == "") {
          webSocket.sendTXT(num, "{\"success\": false, \"message\": \"No game selected\"}");
          return;
        }
        if (!doc.containsKey("command")) {
          webSocket.sendTXT(num, "{\"success\": false, \"message\": \"Missing 'command' in request\"}");
          return;
        }
        String command = doc["command"];
        bool success = false;
        if (currentGame == "ARM") success = play_arm(command);
        else if (currentGame == "SHOOTER") success = play_shooter(command);
        else if (currentGame == "RUBIK") {
          play_robic(command);
          success = true;
        }
        String response = success ? "{\"success\": true, \"message\": \"Command executed\"}" : "{\"success\": false, \"message\": \"Invalid command for " + currentGame + " game\"}";
        webSocket.sendTXT(num, response);
      } else {
        webSocket.sendTXT(num, "{\"success\": false, \"message\": \"Unknown action\"}");
      }
      break;
  }
}

void setup_arm() {
  magnet_state = 0;
  arm_flg = 4;
  arm_servo1.attach(arm_servo1_pin);
  arm_servo2.attach(arm_servo2_pin);
  arm_servo3.attach(arm_servo3_pin);
}

bool play_arm(String command) {
  Serial.println(command);
  static int currentAngle1 = 180, currentAngle2 = 0, currentAngle3 = 0;
  int commas[4];
  commas[0] = command.indexOf(',');
  for (int i = 1; i < 4; i++) commas[i] = command.indexOf(',', commas[i - 1] + 1);
  if (commas[3] != -1) {
    int targetAngle1 = command.substring(0, commas[0]).toInt();
    int targetAngle2 = command.substring(commas[0] + 1, commas[1]).toInt();
    int targetAngle3 = command.substring(commas[1] + 1, commas[2]).toInt();
    magnet_state = command.substring(commas[2] + 1, commas[3]).toInt();
    arm_pickSig = command.substring(commas[3] + 1).toInt();
    if (arm_pickSig) arm_flg = 4;

    targetAngle1 = constrain(targetAngle1, 0, 180);
    targetAngle2 = constrain(targetAngle2, 0, 180);
    targetAngle3 = constrain(targetAngle3, 0, 180);

    digitalWrite(magnetRelayPin, magnet_state ? LOW : HIGH);
    smoothMoveSync(arm_servo1, currentAngle1, targetAngle1,
                   arm_servo2, currentAngle2, targetAngle2,
                   arm_servo3, currentAngle3, targetAngle3, 10, arm_pickSig);
    arm_flg--;
    return true;
  }
  return false;
}

void smoothMoveSync(Servo& s1, int& curr1, int target1, Servo& s2, int& curr2, int target2, Servo& s3, int& curr3, int target3, int delayTime, int pickSig) {
  int diff1 = abs(target1 - curr1), diff2 = abs(target2 - curr2), diff3 = abs(target3 - curr3);
  int maxDiff = max(diff1, max(diff2, diff3));
  if (maxDiff == 0) return;
  if (pickSig % 2 == 0) {
    for (int step = 1; step <= maxDiff; ++step) {
      if (diff2) s2.write(curr2 + ((target2 - curr2) * step) / maxDiff);
      delay(delayTime);
    }
    for (int step = 1; step <= maxDiff; ++step) {
      if (diff1) s1.write(curr1 + ((target1 - curr1) * step) / maxDiff);
      if (diff3) s3.write(curr3 + ((target3 - curr3) * step) / maxDiff);
      delay(delayTime);
    }
  } else {
    for (int step = 1; step <= maxDiff; ++step) {
      if (diff1) s1.write(curr1 + ((target1 - curr1) * step) / maxDiff);
      if (diff3) s3.write(curr3 + ((target3 - curr3) * step) / maxDiff);
      delay(delayTime);
    }
    for (int step = 1; step <= maxDiff; ++step) {
      if (diff2) s2.write(curr2 + ((target2 - curr2) * step) / maxDiff);
      delay(delayTime);
    }
  }
  s1.write(target1);
  s2.write(target2);
  s3.write(target3);
  curr1 = target1;
  curr2 = target2;
  curr3 = target3;
}

void setup_robic() {
  pinMode(robic_step_pin, OUTPUT);
  pinMode(robic_dir_pin, OUTPUT);
  for (int i = 0; i < 5; i++) {
    pinMode(enablePins[i], OUTPUT);
    digitalWrite(enablePins[i], HIGH);
  }
  robic_stepper.setSpeed(MOTOR_RPM);
}

bool play_robic(String command) {
  if (command.startsWith("SOLUTION:")) executeSolution(command.substring(9));
  else executeSolution(command);
  return true;
}

void enableMotor(Motor motor) {
  for (int i = 0; i < 5; i++) digitalWrite(enablePins[i], HIGH);
  digitalWrite(enablePins[motor], LOW);
}

void moveStepper(Motor motor, bool clockwise, int steps) {
  enableMotor(motor);
  for (int i = 0; i < steps; i++) {
    robic_stepper.step(clockwise ? STEPS_PER_90 : -STEPS_PER_90);
    delay(200);
  }
  for (int i = 0; i < 5; i++) digitalWrite(enablePins[i], HIGH);
}

void executeSolution(String solution) {
  for (int i = 0; i < solution.length(); i++) {
    char c = solution.charAt(i);
    Motor motor;
    bool clockwise = true;
    int steps = 1;
    if (c == 'B') motor = BACK;
    else if (c == 'R') motor = RIGHT;
    else if (c == 'L') motor = LEFT;
    else if (c == 'D') motor = DOWN;
    else if (c == 'F') motor = FRONT;
    else continue;
    if (i + 1 < solution.length()) {
      char next = solution.charAt(i + 1);
      if (next == '\'') {
        clockwise = false;
        i++;
      } else if (next == '2') {
        steps = 2;
        i++;
      }
    }
    moveStepper(motor, clockwise, steps);
  }
}

void setup_shooter() {
  shooter_servo1.attach(shooter_servo1_pin,500,2500);
  shooter_servo2.attach(shooter_servo2_pin);
  // pinMode(shooter_step_pin, OUTPUT);
  // pinMode(shooter_dir_pin, OUTPUT);
  // pinMode(shooter_switch_pin, INPUT_PULLUP);
  // digitalWrite(shooter_dir_pin, shooter_reverse_direction);
}

bool play_shooter(String command) {
  if (command.equalsIgnoreCase("shoot")) {
    shoot();
    return true;
  }
  int spaceIndex = command.indexOf(' ');
  if (spaceIndex > 0) {
    int angle1 = command.substring(0, spaceIndex).toInt();
    int angle2 = command.substring(spaceIndex + 1).toInt();
    angle1 = constrain(angle1, minAngle1, maxAngle1);
    angle2 = constrain(angle2, minAngle2, maxAngle2);
    shooter_servo1.write(angle1);
    shooter_servo2.write(angle2);
    return true;
  }
  return false;
}

void shoot() {
  // digitalWrite(shooter_dir_pin, shooter_reverse_direction);
  shooter_flag_home = 0;
  while (!shooter_flag_home) {
    rotateSteps(1);
    // if (digitalRead(shooter_switch_pin) == LOW) {
    //   total_steps = 0;
    //   // digitalWrite(shooter_dir_pin, 1 - shooter_reverse_direction);
    //   rotateSteps(7.2f * 1000);
    //   shooter_flag_home = 1;
    // }
  }
}

void rotateSteps(int steps) {
  for (int i = 0; i < steps; i++) {
    total_steps++;
    // digitalWrite(shooter_step_pin, LOW);
    delayMicroseconds(shooter_motor_speed);
    // digitalWrite(shooter_step_pin, HIGH);
    delayMicroseconds(shooter_motor_speed);
  }
}
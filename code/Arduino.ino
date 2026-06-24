#include <Wire.h>
#include "PCA9685.h"

// Servo driver (PCA9685) ------------------------------------------------------------------------
ServoDriver servo;
static const uint8_t PCA_ADDR = 0x7F;

// Channels for PCA -------------------------------------------------------------------------------
static const uint8_t CH_BASE  = 1; // A  
static const uint8_t CH_WRIST = 2; // B  
static const uint8_t CH_ARM   = 3; // C  
static const uint8_t CH_GRIP  = 4; // D  


//Ultrasoni --------------------------------------------------------------------------------------
const int TRIG_PIN = 9;   // D9
const int ECHO_PIN = 10;  // D10

//Step sizes for motors -------------------------------------------------------------------------------------
static const int STEP_DEG = 5;        // for A/B/C
static const int STEP_DEG_GRIP = 15;  // for D -gripper

//  limits for angles------------------------------------------------------------------------------------------
static const int MIN_DEG = 0;
static const int MAX_DEG = 180;

// Current angles first-----------------------------------------------------------------------------------------------
int degBase  = 90;
int degWrist = 90;
int degArm   = 90;
int degGrip  = 90;

//  flags-------------------------------------------------------------------------------
bool outputsEnabled = false; // send angles after first command
bool busy = false;           // busy = arm is doing an automatic cycle
bool firstAuto = true;       // used for one time startup delay

// Timing  --------------------------------------------------------------------------------
const int STARTUP_WAIT_MS = 2000;      // wait before the first automatic cycle
const int CAMERA_POSE_MS  = 1300;      // hold stable for camera before Pi captures

const int STEP_DELAY_MS_DEFAULT = 80;  // delay between A/B/C steps
const int STEP_DELAY_MS_GRIP    = 110; // delay between D steps

unsigned long lastSend = 0;


// Read one full line from Serial ----------------------------------------------------------
String readLine()
 {
  static String line;
  while (Serial.available())
   {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n') {
      String out = line;
      line = "";
      out.trim();
      return out;
    }
    line += c;
    if (line.length() > 64) { line = ""; return ""; } //  avoid big lines
  }
  return "";
}

// Print angles ------------------------------------------------------------------------------------
void showAngles() {
  Serial.print(F("A=")); Serial.print(degBase);
  Serial.print(F(" B=")); Serial.print(degWrist);
  Serial.print(F(" C=")); Serial.print(degArm);
  Serial.print(F(" D=")); Serial.println(degGrip);
}

// Enable servo outputs once -------------------------------------------------------------------------------------
void enableOutputsIfNeeded() 
{
  if (!outputsEnabled) 
  {
    outputsEnabled = true;
    servo.setAngle(CH_BASE,  degBase);
    servo.setAngle(CH_WRIST, degWrist);
    servo.setAngle(CH_ARM,   degArm);
    servo.setAngle(CH_GRIP,  degGrip);
    delay(100);
  }
}

// Apply a single command ---------------------------------------------------------------------------------------------
void applyCmd(char joint, char op) {
  int step = STEP_DEG;
  if (joint == 'D') step = STEP_DEG_GRIP;         
  int delta = (op == '+') ? step : -step;

  enableOutputsIfNeeded();

  if (joint == 'A') 
  {
    degBase = constrain(degBase + delta, MIN_DEG, MAX_DEG);
    servo.setAngle(CH_BASE, degBase);
  } 
  else if (joint == 'B')
   {
    degWrist = constrain(degWrist + delta, MIN_DEG, MAX_DEG);
    servo.setAngle(CH_WRIST, degWrist);
  } 
  else if (joint == 'C')
   {
    degArm = constrain(degArm + delta, MIN_DEG, MAX_DEG);
    servo.setAngle(CH_ARM, degArm);
  }
   else if (joint == 'D') 
   {
    degGrip = constrain(degGrip + delta, 20, 160);
    servo.setAngle(CH_GRIP, degGrip);
  }
}

// 2-char command--------------------------------------------------------------------------------------------------
void doSteps(const char* cmd2, int times, int delayMs)
 {
  char j = cmd2[0];
  char o = cmd2[1];
  for (int i = 0; i < times; i++)
   {
    applyCmd(j, o);
    delay(delayMs);
  }
}

// Ultrasoni read distance ---------------------------------------------------------------------------------------------
float readDistanceCm() 
{
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // pulseIn timeout 25ms 
  long duration = pulseIn(ECHO_PIN, HIGH, 25000);
  if (duration == 0) return 999.0;

  float cm = duration * 0.0343 / 2.0; // speed of sound -> distance
  return cm;
}

// Clothes are present if distance < 25cm (ignore very big/small numbers)--------------------------------------------------
bool clothesPresent()
 {
  float d = readDistanceCm();
  if (d <= 2.0 || d > 100.0) return false;
  return d < 25.0;
}

// pick + hold for camera ----------------------------------------------------------------------------------------------------
void seq_pick_hold() {
  // move down and forward to reach a cloth
  doSteps("C-", 1, STEP_DELAY_MS_DEFAULT);
  doSteps("C-", 1, STEP_DELAY_MS_DEFAULT);
  doSteps("B+", 1, STEP_DELAY_MS_DEFAULT);

  delay(200); 

  // close the gripper on the cloth
  doSteps("D+", 5, STEP_DELAY_MS_GRIP);

  // return to home pose (camera pose)
  doSteps("B-", 1, STEP_DELAY_MS_DEFAULT);
  doSteps("C+", 1, STEP_DELAY_MS_DEFAULT);
  doSteps("C+", 1, STEP_DELAY_MS_DEFAULT);

  // stay still so the Pi can capture a clean image
  delay(CAMERA_POSE_MS);
}

// drop according to label + return ----------------------------------------------------------------------------------
void seq_drop_and_return(const String& label)
 {
  if (label == "color_shirts") 
  {
    doSteps("A-", 4, STEP_DELAY_MS_DEFAULT); delay(200);
    doSteps("D-", 5, STEP_DELAY_MS_GRIP);    delay(200);
    doSteps("A+", 4, STEP_DELAY_MS_DEFAULT);
  } 
  else if (label == "dresses") 
  {
    doSteps("A-", 7, STEP_DELAY_MS_DEFAULT); delay(200);
    doSteps("D-", 5, STEP_DELAY_MS_GRIP);    delay(200);
    doSteps("A+", 7, STEP_DELAY_MS_DEFAULT);
  } 
  else if (label == "towel") 
  {
    doSteps("A-", 10, STEP_DELAY_MS_DEFAULT); delay(200);
    doSteps("D-", 5, STEP_DELAY_MS_GRIP);     delay(200);
    doSteps("A+", 10, STEP_DELAY_MS_DEFAULT);
  } 
  else if (label == "color_pants")
   {
    doSteps("A+", 4, STEP_DELAY_MS_DEFAULT); delay(200);
    doSteps("D-", 5, STEP_DELAY_MS_GRIP);    delay(200);
    doSteps("A-", 4, STEP_DELAY_MS_DEFAULT);
  } 
  else if (label == "jeans") 
  {
    doSteps("A+", 7, STEP_DELAY_MS_DEFAULT); delay(200);
    doSteps("D-", 5, STEP_DELAY_MS_GRIP);    delay(200);
    doSteps("A-", 7, STEP_DELAY_MS_DEFAULT);
  }
   else if (label == "white")
  {
    doSteps("A+", 10, STEP_DELAY_MS_DEFAULT); delay(200);
    doSteps("D-", 5, STEP_DELAY_MS_GRIP);     delay(200);
    doSteps("A-", 10, STEP_DELAY_MS_DEFAULT);
  } 
  else
  {
    // 
    doSteps("D-", 5, STEP_DELAY_MS_GRIP);
  }

  delay(200);
}

// Manual control (A+/A-/B+/B-/C+/C-/D+/D-) for calibration--------------------------------------------------------------------
void handleManual2Char(const String& cmd)
 {
  if (cmd.length() != 2) return;
  char joint = toupper(cmd[0]);
  char op    = cmd[1];

  if ((joint=='A'||joint=='B'||joint=='C'||joint=='D') && (op==
'+'||op=='-'))
   {
    applyCmd(joint, op);
    showAngles(); 
  }
}

void setup()
 {
  Serial.begin(115200);

  Wire.begin();
  Wire.setClock(400000); 

  servo.init(PCA_ADDR);
  delay(100);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  Serial.println("READY");
  showAngles();
}

void loop()
 {
  // If we are not busy, we report "FULL" when clothes are present.
  // (Pi listens for "FULL", then sends "PICK")
  if (!busy)
   {
    bool present = clothesPresent();

    // simple rate-limit
    if (present && millis() - lastSend > 400) 
    {
      Serial.println("FULL");
      lastSend = millis();
    }
  }

  // Read commands from Pi (or manual commands from Serial Monitor)
  String line = readLine();
  if (line.length() == 0) return;
  line.trim();

  // Manual control always available 
  if (line.length() == 2) 
  {
    handleManual2Char(line);
    return;
  }

  String upper = line;
  upper.toUpperCase();

  // Pi says "PICK" - arm picks and then prints HOLD (camera pose ready)
  if (upper == "PICK")
   {
    if (firstAuto)
     {
       delay(STARTUP_WAIT_MS); firstAuto = false;
     }
    busy = true;
    seq_pick_hold();
    Serial.println("HOLD");
    return;
  }

  // Pi says DROP <label> - drop in correct basket -print "DONE"
  if (upper.startsWith("DROP ")) 
  {
    String label = line.substring(5);
    label.toLowerCase();

    // handshake - tell Pi we got the label
    Serial.print("ACK ");
    Serial.println(label);
-
    seq_drop_and_return(label);
    busy = false;
    Serial.println("DONE");
    return;
  }
}

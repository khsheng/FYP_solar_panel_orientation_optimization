#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ESP32Servo.h>
#include <OneWire.h>
#include <DallasTemperature.h>

// Pin Definitions
#define LED 4
#define SERVO_PIN 13
#define OUT_PUMP 27
#define IN_PUMP 26
#define TEMP_PIN 32        // DS18B20 data pin
#define LDR_PIN 33
#define BUZZER 5
#define BUTTON 25

// Ultrasonic
#define TRIG_PIN 18
#define ECHO_PIN 19

// Temperature thresholds
#define TEMP_MIN 22
#define TEMP_MAX 32

// Water level thresholds (in cm, measured by ultrasonic)
#define WATER_FULL_DIST 2       // distance from sensor to water when tank is full
#define WATER_EMPTY_DIST 7     // distance from sensor to water when tank is empty

// LCD
LiquidCrystal_I2C lcd(0x27, 16, 2);

// Servo
Servo myServo;

// DS18B20
OneWire oneWire(TEMP_PIN);
DallasTemperature sensors(&oneWire);

// Variables
int ldrValue = 0;
long duration;
float distance;
float temperature = 0;

// Buzzer alarm timing
unsigned long previousMillis = 0;
const long interval = 500; // 500ms ON/OFF
bool buzzerState = false;

void setup() {
  Serial.begin(115200);

  pinMode(LED, OUTPUT);
  pinMode(OUT_PUMP, OUTPUT);
  pinMode(IN_PUMP, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  pinMode(BUTTON, INPUT_PULLUP);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  myServo.setPeriodHertz(50);
  myServo.attach(SERVO_PIN, 500, 2400);
  myServo.write(0);

  sensors.begin();

  lcd.init();
  lcd.backlight();

  lcd.setCursor(0, 0);
  lcd.print("System Starting");
  delay(2000);
  lcd.clear();
}

void loop() {
  // ===== ULTRASONIC READ =====
  distance = readUltrasonic();

  // ===== TEMP READ (DS18B20) =====
  sensors.requestTemperatures();
  temperature = sensors.getTempCByIndex(0);

  // ===== LDR READ =====
  ldrValue = analogRead(LDR_PIN);
  Serial.println("LDR Reading: " + String(ldrValue));

  // ===== DISPLAY =====
  lcd.setCursor(0, 0);
  lcd.print("T:");
  lcd.print(temperature);
  lcd.print((char)223); // degree symbol
  lcd.print("C ");

  lcd.setCursor(0, 1);
  lcd.print("D:");
  lcd.print(distance);
  lcd.print("cm ");

  // ===== LED (LIGHT CONTROL) =====
  digitalWrite(LED, (ldrValue > 1888) ? HIGH : LOW);

  // ===== TEMPERATURE ALERT =====
  if (temperature < TEMP_MIN || temperature > TEMP_MAX) {
    alarmBuzzer();
  } else {
    digitalWrite(BUZZER, LOW); // turn off buzzer
    buzzerState = false;
    previousMillis = millis();
  }

  // ===== BUTTON: initiate water change =====
  if (digitalRead(BUTTON) == LOW) {
    myServo.write(90);
    delay(500); // give the servo time to turn
    waterChangeSequence();
    myServo.write(0); // return servo to 0 degrees
  }

  delay(500);
}

// ===== ULTRASONIC READ FUNCTION =====
float readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH);
  float dist = duration * 0.034 / 2;
  return dist;
}

// ===== ALARM BUZZER FUNCTION =====
void alarmBuzzer() {
  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;
    buzzerState = !buzzerState;
    digitalWrite(BUZZER, buzzerState ? HIGH : LOW);
  }
}

// ===== WATER CHANGE FUNCTION =====
void waterChangeSequence() {
  lcd.clear();
  Serial.println("Starting Water Change...");

  // ===== DRAINING =====
  lcd.setCursor(0,0);
  lcd.print("Draining Water  ");

  digitalWrite(OUT_PUMP, HIGH);
  digitalWrite(IN_PUMP, LOW);

  // Wait until water reaches empty threshold
  while (readUltrasonic() < WATER_EMPTY_DIST) {
    distance = readUltrasonic();
    lcd.setCursor(0,1);
    lcd.print("D:");
    lcd.print(distance);
    lcd.print("cm   ");
    delay(500);
  }
  digitalWrite(OUT_PUMP, LOW); // Stop draining
  Serial.println("Water Drained");
  lcd.setCursor(0,0);
  lcd.clear();
  lcd.print("Water Drained");
  delay(2000);
  // ===== REFILLING =====
  lcd.setCursor(0,0);
  lcd.print("Refilling Water");

  digitalWrite(IN_PUMP, HIGH);
  digitalWrite(OUT_PUMP, LOW);

  // Wait until water reaches full threshold
  while (readUltrasonic() > WATER_FULL_DIST) {
    distance = readUltrasonic();
    lcd.setCursor(0,1);
    lcd.print("D:");
    lcd.print(distance);
    lcd.print("cm   ");
    delay(500);
  }
  digitalWrite(IN_PUMP, LOW); // Stop refilling
  Serial.println("Water Refilled");

  lcd.setCursor(0,1);
  lcd.print("Water Change OK");
  delay(2000);
  lcd.clear();
}
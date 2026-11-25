// Generic Arduino-ESP32 (v3.x) Wake-Word Example for tiny-ja-wakeword
// Pure C++17, zero TFLite / zero Python / zero external dictionary (~525 KB Flash, ~88.5 KB SRAM).
//
// Works on ESP32-S3 class boards (Flash >= 1 MB, SRAM >= 256 KB, 240 MHz recommended).
// Supports both onboard/external PDM microphones and standard I2S microphones (e.g., INMP441).
//
// Setup:
//   1. Copy the 5 headers from `cpp/` (fbank.hpp, conv_model.hpp, weights_conv_int8.h, kana.hpp, aligner.hpp)
//      into this sketch folder (or your Arduino `libraries/tiny_ja_wakeword/src/` directory).
//   2. Adjust the microphone type and GPIO pins below to match your board.

#include <Arduino.h>
#include <ESP_I2S.h>

#include "aligner.hpp"
#include "conv_model.hpp"
#include "fbank.hpp"
#include "kana.hpp"

using namespace tiny_ja_ww;

// --- 1. Wake-word & threshold settings ---
// Specify a UTF-8 hiragana or katakana phrase (6+ phonemes recommended).
static const char* kWakeWord = "オーケー、グーグル";
static constexpr float kThreshold = 0.50f;  // 0.30=sensitive, 0.50=balanced, 0.70=strict

// --- 2. Microphone wiring settings (adjust to your board) ---
// Set to 1 for a PDM microphone (CLK + DATA), or 0 for a standard I2S microphone (BCLK + WS + DIN, e.g. INMP441).
#define USE_PDM_MIC 1

#if USE_PDM_MIC
static constexpr int kPdmClkPin  = 42;
static constexpr int kPdmDataPin = 41;
#else
static constexpr int kI2sBclkPin = 5;
static constexpr int kI2sWsPin   = 6;
static constexpr int kI2sDinPin  = 4;
#endif

// Optional status LED (set to -1 to disable)
static constexpr int kLedPin = -1;
static constexpr bool kLedActiveLow = true;

static constexpr int kSampleRate = 16000;
static constexpr int kChunkSamples = 640;  // 40 ms = 1 CTC output frame

I2SClass I2S;
static KaldiFbank80 g_fbank;
static StreamingConvInt8Model g_model;
static WakeWordAligner g_aligner;

static int16_t g_pcm[kChunkSamples];
static float g_audio[kChunkSamples];
static uint32_t g_led_until_ms = 0;

static void set_led(bool on) {
  if (kLedPin < 0) return;
  digitalWrite(kLedPin, (on ^ kLedActiveLow) ? HIGH : LOW);
}

void setup() {
  Serial.begin(115200);
  if (kLedPin >= 0) {
    pinMode(kLedPin, OUTPUT);
    set_led(false);
  }

  std::vector<int> tokens;
  if (!kana_to_token_ids(kWakeWord, tokens) || !g_aligner.enroll(tokens, kThreshold)) {
    Serial.println("ERROR: Invalid wake-word (use UTF-8 hiragana or katakana).");
    while (true) delay(1000);
  }

#if USE_PDM_MIC
  I2S.setPinsPdmRx(kPdmClkPin, kPdmDataPin);
  bool ok = I2S.begin(I2S_MODE_PDM_RX, kSampleRate, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
#else
  I2S.setPins(kI2sBclkPin, kI2sWsPin, -1, kI2sDinPin);
  bool ok = I2S.begin(I2S_MODE_STD, kSampleRate, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO);
#endif
  if (!ok) {
    Serial.println("ERROR: Failed to initialize I2S/PDM microphone. Check pin numbers.");
    while (true) delay(1000);
  }

  Serial.printf("tiny-ja-wakeword ready! phrase=%s (%u phones), threshold=%.0f%%, sram=%u B\n",
                kWakeWord, static_cast<unsigned>(tokens.size()), kThreshold * 100.0f,
                static_cast<unsigned>(sizeof(g_fbank) + sizeof(g_model) + sizeof(g_aligner)));
}

void loop() {
  size_t bytes_read = I2S.readBytes(reinterpret_cast<char*>(g_pcm), sizeof(g_pcm));
  int n_samples = static_cast<int>(bytes_read / sizeof(int16_t));
  if (n_samples <= 0) return;

  for (int i = 0; i < n_samples; ++i) {
    g_audio[i] = static_cast<float>(g_pcm[i]) / 32768.0f;
  }

  g_fbank.accept(g_audio, n_samples, [](const float fb[80]) {
    float logp[44];
    if (g_model.push_fbank(fb, logp)) {
      DetectionEvent ev;
      if (g_aligner.push(logp, &ev)) {
        Serial.printf("[DETECTED #%d] %s  prob=%.1f%%  t=%.2fs\n",
                      ev.call, kWakeWord, ev.probability * 100.0f, ev.end_seconds);
        set_led(true);
        g_led_until_ms = millis() + 1500;
      }
    }
  });

  if (g_led_until_ms != 0 && millis() >= g_led_until_ms) {
    set_led(false);
    g_led_until_ms = 0;
  }
}

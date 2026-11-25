# tiny-ja-wakeword 純 C++17 組み込み推論エンジン

Python・ONNX Runtime・TFLite Micro・外部辞書を一切使わず、**標準 C++17 ヘッダ 5 本だけ** で `conv` モデル（8 ビット量子化）をストリーミング実行するための組み込み実装です。

## ファイル構成

| ファイル | 役割 | メモリ規模 |
|---|---|---|
| `fbank.hpp` | 16 kHz 音声 → 80 次元 Kaldi 互換周波数特徴（25 ms 窓・10 ms 送り・512 点 FFT） | 約 10 KB |
| `conv_model.hpp` | 40 ms 刻みストリーミング 8 ビット畳み込み推論（ヒープ確保ゼロ） | 約 70 KB |
| `weights_conv_int8.h` | `conv` モデルの 8 ビット重み＋0〜100% 確率校正係数（Flash `.rodata` 配置） | **Flash 525.4 KB** |
| `kana.hpp` | 外部辞書なしの UTF-8 ひらがな・カタカナ → 44 音素 ID 変換 | 数 KB |
| `aligner.hpp` | 自由開始 CTC 整列＋4 特徴 0〜100% 確率判定＋重複再発火抑止 | 約 8 KB |
| `bench_cli.cpp` | PC / Linux ボード向けの単体 WAV テスト用 CLI | — |
| `examples/arduino_esp32/arduino_esp32.ino` | Arduino-ESP32（v3 系）向け汎用マイク検出スケッチ（PDM / 標準 I2S 両対応） | — |

総作業メモリ（`KaldiFbank80 + StreamingConvInt8Model + WakeWordAligner`）は **`90,604 B`（約 88.5 KB）** です。

## 動作要件の目安

- **必要メモリ**：Flash 1 MB 以上・SRAM 256 KB 以上（重み 525 KB・作業領域 88.5 KB）
- **対象ハードウェア例**：
  - ESP32-S3 系開発ボード全般（内蔵 SRAM 512 KB 内に収まり、PSRAM なしでも配置可能）
  - Teensy 4.x / STM32H7 / Raspberry Pi Pico 2（RP2350）等の高速 32 ビットマイコン（マイク録音部のみ各ボードのドライバへ差し替え）
  - Raspberry Pi・Linux SBC・PC（そのままネイティブビルド可能）
- **非対応**：Arduino Uno / Nano / Mega などの AVR 系マイコン（Flash 32〜256 KB・SRAM 2〜8 KB のため収まりません）

## PC / Raspberry Pi での単体ビルド例

外部ライブラリなしでそのままコンパイルできます。

```bash
# Linux / macOS / Raspberry Pi
g++ -std=c++17 -O3 -I cpp cpp/bench_cli.cpp -o bench_cli
./bench_cli sample_16k_mono.wav "オーケー、グーグル" 0.50

# Windows (MinGW / zig c++)
g++ -std=c++17 -O3 -I cpp cpp/bench_cli.cpp -lshell32 -o bench_cli.exe
```

## Arduino-ESP32 での使い方

1. `cpp/` 内の 5 ヘッダ（`fbank.hpp`・`conv_model.hpp`・`weights_conv_int8.h`・`kana.hpp`・`aligner.hpp`）を `examples/arduino_esp32/` フォルダ内へコピーします。
2. `arduino_esp32.ino` を Arduino IDE（`esp32` コア v3 系）で開き、先頭の `kWakeWord`・`USE_PDM_MIC`・マイクの GPIO ピン番号をお使いのボードに合わせて設定して書き込みます。

> **ESP32-S3 での高速化ヒント**：全積和演算の約 9 割は `conv_model.hpp` 内の `dot_i8(a, b, n)`（長さ `n` は常に 16 の倍数の `224` または `320`）に集中しています。実機でさらに CPU 負荷を下げたい場合は、この 1 関数だけを ESP32-S3 固有の 8 ビット SIMD 命令（ESP-NN / PIE）へ差し替えるだけで全体が高速化します。

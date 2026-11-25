// Standalone zero-dependency C++17 CLI to verify fbank.hpp + conv_model.hpp + aligner.hpp + kana.hpp on PC.
// Usage:
//   bench_cli <wav_16k_mono> <kana_phrase> [threshold=0.5] [--dump-fbank out.f32] [--dump-logp out.f32]
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

#include "aligner.hpp"
#include "conv_model.hpp"
#include "fbank.hpp"
#include "kana.hpp"

using namespace tiny_ja_ww;

static std::vector<std::string> get_utf8_args(int argc, char** argv) {
  std::vector<std::string> out;
#ifdef _WIN32
  int wargc = 0;
  LPWSTR* wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
  if (wargv) {
    for (int i = 0; i < wargc; ++i) {
      int len = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
      std::string s(len > 0 ? len - 1 : 0, '\0');
      if (len > 1) WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], len, nullptr, nullptr);
      out.push_back(std::move(s));
    }
    LocalFree(wargv);
    return out;
  }
#endif
  for (int i = 0; i < argc; ++i) out.emplace_back(argv[i]);
  return out;
}

static bool read_wav_16k_pcm16(const std::string& path, std::vector<float>& samples) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char hdr[12];
  f.read(hdr, 12);
  if (std::memcmp(hdr, "RIFF", 4) != 0 || std::memcmp(hdr + 8, "WAVE", 4) != 0) return false;
  while (f) {
    char cid[4];
    uint32_t csize = 0;
    f.read(cid, 4);
    f.read(reinterpret_cast<char*>(&csize), 4);
    if (!f) break;
    if (std::memcmp(cid, "data", 4) == 0) {
      size_t n = csize / 2;
      std::vector<int16_t> pcm(n);
      f.read(reinterpret_cast<char*>(pcm.data()), csize);
      samples.resize(n);
      for (size_t i = 0; i < n; ++i) samples[i] = static_cast<float>(pcm[i]) / 32768.0f;
      return true;
    }
    f.seekg(csize + (csize & 1), std::ios::cur);
  }
  return false;
}

int main(int argc, char** argv) {
  auto args = get_utf8_args(argc, argv);
  if (args.size() < 3) {
    std::cerr << "Usage: bench_cli <wav_16k_pcm16> <kana_phrase> [threshold] [--dump-fbank path] [--dump-logp path]\n";
    return 1;
  }
  std::string wav_path = args[1];
  std::string phrase = args[2];
  float threshold = (args.size() >= 4 && args[3][0] != '-') ? std::strtof(args[3].c_str(), nullptr) : 0.50f;
  std::string dump_fb, dump_lp;
  for (size_t i = 3; i < args.size(); ++i) {
    if (args[i] == "--dump-fbank" && i + 1 < args.size()) dump_fb = args[++i];
    if (args[i] == "--dump-logp" && i + 1 < args.size()) dump_lp = args[++i];
  }

  std::vector<int> tokens;
  if (!kana_to_token_ids(phrase, tokens)) {
    std::cerr << "Failed to convert kana phrase to tokens.\n";
    return 2;
  }
  std::vector<float> audio;
  if (!read_wav_16k_pcm16(wav_path, audio)) {
    std::cerr << "Failed to read 16kHz PCM16 WAV: " << wav_path << "\n";
    return 3;
  }

  KaldiFbank80 fbank;
  StreamingConvInt8Model model;
  WakeWordAligner aligner;
  aligner.enroll(tokens, threshold);

  std::ofstream fb_out, lp_out;
  if (!dump_fb.empty()) fb_out.open(dump_fb, std::ios::binary);
  if (!dump_lp.empty()) lp_out.open(dump_lp, std::ios::binary);

  int n_fb = 0, n_lp = 0, n_events = 0;
  auto t0 = std::chrono::steady_clock::now();

  auto handle_fb = [&](const float fb[80]) {
    n_fb++;
    if (fb_out) fb_out.write(reinterpret_cast<const char*>(fb), sizeof(float) * 80);
    float logp[44];
    if (model.push_fbank(fb, logp)) {
      n_lp++;
      if (lp_out) lp_out.write(reinterpret_cast<const char*>(logp), sizeof(float) * 44);
      DetectionEvent ev;
      if (aligner.push(logp, &ev)) {
        n_events++;
        std::printf("EVENT call=%d prob=%.4f start=%.2fs end=%.2fs\n",
                    ev.call, ev.probability, ev.start_frame * 0.04f, ev.end_seconds);
      }
    }
  };

  // Stream in 1600-sample (100ms) chunks just like a real microphone
  for (size_t pos = 0; pos < audio.size(); pos += 1600) {
    int n = static_cast<int>(std::min<size_t>(1600, audio.size() - pos));
    fbank.accept(audio.data() + pos, n, handle_fb);
  }
  fbank.finish(handle_fb);

  auto t1 = std::chrono::steady_clock::now();
  double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  double audio_s = static_cast<double>(audio.size()) / 16000.0;
  std::printf("SUMMARY audio=%.2fs fbank_frames=%d ctc_frames=%d events=%d elapsed=%.1fms (RTF=%.3f%%, 10s=%.1fms) sram_model=%zuB\n",
              audio_s, n_fb, n_lp, n_events, elapsed_ms, (elapsed_ms / 1000.0) / audio_s * 100.0,
              elapsed_ms / audio_s * 10.0, sizeof(StreamingConvInt8Model) + sizeof(KaldiFbank80) + sizeof(WakeWordAligner));
  return 0;
}

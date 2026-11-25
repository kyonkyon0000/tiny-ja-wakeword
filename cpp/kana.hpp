// Zero-dictionary UTF-8 Japanese hiragana/katakana -> 44-token CTC phoneme ID converter in pure C++17.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tiny_ja_ww {

// Token IDs matching the 44-token model vocabulary (<blk>=0 .. z=43)
enum TokenId : int {
  TOK_BLK = 0, TOK_UNK = 1, TOK_I = 2, TOK_N = 3, TOK_U = 4,
  TOK_a = 5, TOK_b = 6, TOK_by = 7, TOK_ch = 8, TOK_cl = 9,
  TOK_d = 10, TOK_dy = 11, TOK_e = 12, TOK_f = 13, TOK_fy = 14,
  TOK_g = 15, TOK_gw = 16, TOK_gy = 17, TOK_h = 18, TOK_hy = 19,
  TOK_i = 20, TOK_j = 21, TOK_k = 22, TOK_kw = 23, TOK_ky = 24,
  TOK_m = 25, TOK_my = 26, TOK_n = 27, TOK_ny = 28, TOK_o = 29,
  TOK_p = 30, TOK_py = 31, TOK_r = 32, TOK_ry = 33, TOK_s = 34,
  TOK_sh = 35, TOK_t = 36, TOK_ts = 37, TOK_ty = 38, TOK_u = 39,
  TOK_v = 40, TOK_w = 41, TOK_y = 42, TOK_z = 43,
};

inline bool is_vowel_token(int tok) {
  return tok == TOK_a || tok == TOK_i || tok == TOK_u || tok == TOK_e || tok == TOK_o;
}

inline std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
  std::vector<uint32_t> cps;
  size_t i = 0, n = s.size();
  while (i < n) {
    uint8_t c = static_cast<uint8_t>(s[i]);
    if (c < 0x80) {
      cps.push_back(c);
      i += 1;
    } else if ((c & 0xE0) == 0xC0 && i + 1 < n) {
      cps.push_back(((c & 0x1F) << 6) | (static_cast<uint8_t>(s[i + 1]) & 0x3F));
      i += 2;
    } else if ((c & 0xF0) == 0xE0 && i + 2 < n) {
      cps.push_back(((c & 0x0F) << 12) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 6) |
                    (static_cast<uint8_t>(s[i + 2]) & 0x3F));
      i += 3;
    } else if ((c & 0xF8) == 0xF0 && i + 3 < n) {
      cps.push_back(((c & 0x07) << 18) | ((static_cast<uint8_t>(s[i + 1]) & 0x3F) << 12) |
                    ((static_cast<uint8_t>(s[i + 2]) & 0x3F) << 6) | (static_cast<uint8_t>(s[i + 3]) & 0x3F));
      i += 4;
    } else {
      i += 1;
    }
  }
  return cps;
}

inline bool is_skip_cp(uint32_t cp) {
  return cp == ' ' || cp == '\t' || cp == '\r' || cp == '\n' || cp == 0x3000 ||
         cp == 0x3001 || cp == 0x3002 || cp == ',' || cp == '.' || cp == '!' || cp == '?' ||
         cp == 0xFF01 || cp == 0xFF1F || cp == 0x30FB || cp == '-' || cp == 0x2010 || cp == 0x2015 ||
         cp == 0x300C || cp == 0x300D || cp == '(' || cp == ')' || cp == '~' || cp == 0x301C;
}

// Convert UTF-8 hiragana/katakana string to phoneme token IDs. Returns false if unsupported chars exist.
inline bool kana_to_token_ids(const std::string& utf8_text, std::vector<int>& out) {
  out.clear();
  auto raw = utf8_to_codepoints(utf8_text);
  // Normalize katakana (U+30A1..U+30F6) -> hiragana (U+3041..U+3096)
  std::vector<uint32_t> hira;
  hira.reserve(raw.size());
  for (uint32_t cp : raw) {
    if (cp >= 0x30A1 && cp <= 0x30F6) cp -= 0x60;
    hira.push_back(cp);
  }

  auto push2 = [&](int a, int b) { out.push_back(a); out.push_back(b); };
  auto match_digraph = [&](uint32_t a, uint32_t b) -> bool {
    // Small ya/yu/yo/e/a/i/u/o: ゃ=3083, ゅ=3085, ょ=3087, ぁ=3041, ぃ=3043, ぅ=3045, ぇ=3047, ぉ=3049
    int v = -1;
    if (b == 0x3083 || b == 0x3041) v = TOK_a;
    else if (b == 0x3043) v = TOK_i;
    else if (b == 0x3085 || b == 0x3045) v = TOK_u;
    else if (b == 0x3047) v = TOK_e;
    else if (b == 0x3087 || b == 0x3049) v = TOK_o;
    if (v < 0) return false;

    bool small_y = (b == 0x3083 || b == 0x3085 || b == 0x3087 || b == 0x3047);
    if (small_y) {
      switch (a) {
        case 0x304D: push2(TOK_ky, v); return true;  // き
        case 0x304E: push2(TOK_gy, v); return true;  // ぎ
        case 0x3057: push2(TOK_sh, v); return true;  // し
        case 0x3058: case 0x3062: push2(TOK_j, v); return true;  // じ・ぢ
        case 0x3061: push2(TOK_ch, v); return true;  // ち
        case 0x306B: push2(TOK_ny, v); return true;  // に
        case 0x3072: push2(TOK_hy, v); return true;  // ひ
        case 0x3073: push2(TOK_by, v); return true;  // び
        case 0x3074: push2(TOK_py, v); return true;  // ぴ
        case 0x307F: push2(TOK_my, v); return true;  // み
        case 0x308A: push2(TOK_ry, v); return true;  // り
        default: break;
      }
    }
    if (a == 0x3075) {  // ふ
      if (b == 0x3085 || b == 0x3087) { push2(TOK_fy, v); return true; }
      if (b == 0x3041 || b == 0x3043 || b == 0x3047 || b == 0x3049) { push2(TOK_f, v); return true; }
    }
    if (a == 0x3066) {  // て
      if (b == 0x3043) { push2(TOK_t, TOK_i); return true; }
      if (b == 0x3085) { push2(TOK_ty, TOK_u); return true; }
    }
    if (a == 0x3067) {  // で
      if (b == 0x3043) { push2(TOK_d, TOK_i); return true; }
      if (b == 0x3085) { push2(TOK_dy, TOK_u); return true; }
    }
    if (a == 0x3068 && b == 0x3045) { push2(TOK_t, TOK_u); return true; }  // とぅ
    if (a == 0x3069 && b == 0x3045) { push2(TOK_d, TOK_u); return true; }  // どぅ
    if (a == 0x3064 && (b == 0x3041 || b == 0x3043 || b == 0x3047 || b == 0x3049)) { push2(TOK_ts, v); return true; }
    if (a == 0x3046 && (b == 0x3043 || b == 0x3047 || b == 0x3049)) { push2(TOK_w, v); return true; }
    if (a == 0x3094) {  // ゔ
      if (b == 0x3085) { push2(TOK_by, TOK_u); return true; }
      if (b == 0x3041 || b == 0x3043 || b == 0x3047 || b == 0x3049) { push2(TOK_v, v); return true; }
    }
    if (a == 0x304F && (b == 0x3041 || b == 0x3043 || b == 0x3047 || b == 0x3049)) { push2(TOK_kw, v); return true; }
    if (a == 0x3050 && (b == 0x3041 || b == 0x3043 || b == 0x3047 || b == 0x3049)) { push2(TOK_gw, v); return true; }
    if (a == 0x3059 && b == 0x3043) { push2(TOK_s, TOK_i); return true; }
    if (a == 0x305A && b == 0x3043) { push2(TOK_z, TOK_i); return true; }
    return false;
  };

  bool prev_break = true;
  for (size_t i = 0; i < hira.size(); ) {
    uint32_t ch = hira[i];
    if (is_skip_cp(ch)) { prev_break = true; ++i; continue; }
    if (ch == 0x30FC) {  // ー
      if (!out.empty() && is_vowel_token(out.back())) out.push_back(out.back());
      ++i; continue;
    }
    if (i + 1 < hira.size() && match_digraph(ch, hira[i + 1])) {
      prev_break = false; i += 2; continue;
    }
    // Single hiragana mapping
    switch (ch) {
      case 0x3042: case 0x3041: out.push_back(TOK_a); break;
      case 0x3044: case 0x3043: case 0x3090: out.push_back(TOK_i); break;
      case 0x3046:
        if (!prev_break && !out.empty() && out.back() == TOK_o) out.push_back(TOK_o);
        else out.push_back(TOK_u);
        break;
      case 0x3045: out.push_back(TOK_u); break;
      case 0x3048: case 0x3047: case 0x3091: out.push_back(TOK_e); break;
      case 0x304A: case 0x3049: case 0x3092: out.push_back(TOK_o); break;
      case 0x304B: push2(TOK_k, TOK_a); break; case 0x304D: push2(TOK_k, TOK_i); break;
      case 0x304F: push2(TOK_k, TOK_u); break; case 0x3051: push2(TOK_k, TOK_e); break;
      case 0x3053: push2(TOK_k, TOK_o); break;
      case 0x304C: push2(TOK_g, TOK_a); break; case 0x304E: push2(TOK_g, TOK_i); break;
      case 0x3050: push2(TOK_g, TOK_u); break; case 0x3052: push2(TOK_g, TOK_e); break;
      case 0x3054: push2(TOK_g, TOK_o); break;
      case 0x3055: push2(TOK_s, TOK_a); break; case 0x3057: push2(TOK_sh, TOK_i); break;
      case 0x3059: push2(TOK_s, TOK_u); break; case 0x305B: push2(TOK_s, TOK_e); break;
      case 0x305D: push2(TOK_s, TOK_o); break;
      case 0x3056: push2(TOK_z, TOK_a); break; case 0x3058: case 0x3062: push2(TOK_j, TOK_i); break;
      case 0x305A: case 0x3065: push2(TOK_z, TOK_u); break;
      case 0x305C: push2(TOK_z, TOK_e); break; case 0x305E: push2(TOK_z, TOK_o); break;
      case 0x305F: push2(TOK_t, TOK_a); break; case 0x3061: push2(TOK_ch, TOK_i); break;
      case 0x3064: push2(TOK_ts, TOK_u); break; case 0x3066: push2(TOK_t, TOK_e); break;
      case 0x3068: push2(TOK_t, TOK_o); break;
      case 0x3060: push2(TOK_d, TOK_a); break; case 0x3067: push2(TOK_d, TOK_e); break;
      case 0x3069: push2(TOK_d, TOK_o); break;
      case 0x306A: push2(TOK_n, TOK_a); break; case 0x306B: push2(TOK_n, TOK_i); break;
      case 0x306C: push2(TOK_n, TOK_u); break; case 0x306D: push2(TOK_n, TOK_e); break;
      case 0x306E: push2(TOK_n, TOK_o); break;
      case 0x306F: push2(TOK_h, TOK_a); break; case 0x3072: push2(TOK_h, TOK_i); break;
      case 0x3075: push2(TOK_f, TOK_u); break; case 0x3078: push2(TOK_h, TOK_e); break;
      case 0x307B: push2(TOK_h, TOK_o); break;
      case 0x3070: push2(TOK_b, TOK_a); break; case 0x3073: push2(TOK_b, TOK_i); break;
      case 0x3076: push2(TOK_b, TOK_u); break; case 0x3079: push2(TOK_b, TOK_e); break;
      case 0x307C: push2(TOK_b, TOK_o); break;
      case 0x3071: push2(TOK_p, TOK_a); break; case 0x3074: push2(TOK_p, TOK_i); break;
      case 0x3077: push2(TOK_p, TOK_u); break; case 0x307A: push2(TOK_p, TOK_e); break;
      case 0x307D: push2(TOK_p, TOK_o); break;
      case 0x307E: push2(TOK_m, TOK_a); break; case 0x307F: push2(TOK_m, TOK_i); break;
      case 0x3080: push2(TOK_m, TOK_u); break; case 0x3081: push2(TOK_m, TOK_e); break;
      case 0x3082: push2(TOK_m, TOK_o); break;
      case 0x3084: case 0x3083: push2(TOK_y, TOK_a); break;
      case 0x3086: case 0x3085: push2(TOK_y, TOK_u); break;
      case 0x3088: case 0x3087: push2(TOK_y, TOK_o); break;
      case 0x3089: push2(TOK_r, TOK_a); break; case 0x308A: push2(TOK_r, TOK_i); break;
      case 0x308B: push2(TOK_r, TOK_u); break; case 0x308C: push2(TOK_r, TOK_e); break;
      case 0x308D: push2(TOK_r, TOK_o); break;
      case 0x308F: case 0x308E: push2(TOK_w, TOK_a); break;
      case 0x3093: out.push_back(TOK_N); break;
      case 0x3063: out.push_back(TOK_cl); break;
      case 0x3094: push2(TOK_v, TOK_u); break;
      default: return false;
    }
    prev_break = false;
    ++i;
  }
  return !out.empty();
}

}  // namespace tiny_ja_ww

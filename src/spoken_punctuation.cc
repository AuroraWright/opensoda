#include "spoken_punctuation.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

namespace soda {
namespace {

// A text-format protobuf message: scalar fields as strings, in order.
struct TextMessage {
  std::vector<std::pair<std::string, std::string>> scalars;
  std::vector<std::pair<std::string, TextMessage>> messages;
  std::string Get(const std::string& name) const {
    for (const auto& [k, v] : scalars) {
      if (k == name) return v;
    }
    return "";
  }
};

class TextParser {
 public:
  explicit TextParser(const std::string& s) : s_(s) {}
  bool Parse(TextMessage* m) { return ParseFields(m) && pos_ == s_.size(); }

 private:
  void Skip() {
    while (pos_ < s_.size()) {
      if (std::isspace(static_cast<unsigned char>(s_[pos_]))) {
        ++pos_;
      } else if (s_[pos_] == '#') {
        while (pos_ < s_.size() && s_[pos_] != '\n') ++pos_;
      } else {
        break;
      }
    }
  }
  bool ParseFields(TextMessage* m) {
    for (Skip(); pos_ < s_.size() && s_[pos_] != '}'; Skip()) {
      size_t start = pos_;
      while (pos_ < s_.size() && (std::isalnum(static_cast<unsigned char>(s_[pos_])) || s_[pos_] == '_')) ++pos_;
      if (pos_ == start) return false;
      std::string name = s_.substr(start, pos_ - start);
      Skip();
      if (pos_ < s_.size() && s_[pos_] == ':') {
        ++pos_;
        Skip();
      }
      if (pos_ < s_.size() && s_[pos_] == '{') {
        ++pos_;
        TextMessage sub;
        if (!ParseFields(&sub)) return false;
        Skip();
        if (pos_ >= s_.size() || s_[pos_] != '}') return false;
        ++pos_;
        m->messages.emplace_back(std::move(name), std::move(sub));
      } else if (pos_ < s_.size() && s_[pos_] == '"') {
        std::string value;
        if (!ParseString(&value)) return false;
        m->scalars.emplace_back(std::move(name), std::move(value));
      } else {
        start = pos_;
        while (pos_ < s_.size() && !std::isspace(static_cast<unsigned char>(s_[pos_])) && s_[pos_] != '}') ++pos_;
        m->scalars.emplace_back(std::move(name), s_.substr(start, pos_ - start));
      }
    }
    return true;
  }
  bool ParseString(std::string* out) {
    ++pos_;
    while (pos_ < s_.size() && s_[pos_] != '"') {
      char c = s_[pos_++];
      if (c != '\\') {
        *out += c;
        continue;
      }
      if (pos_ >= s_.size()) return false;
      c = s_[pos_++];
      if (c >= '0' && c <= '7') {
        int v = c - '0';
        for (int k = 0; k < 2 && pos_ < s_.size() && s_[pos_] >= '0' && s_[pos_] <= '7'; ++k) v = v * 8 + (s_[pos_++] - '0');
        *out += static_cast<char>(v);
      } else {
        *out += c == 'n' ? '\n' : c == 't' ? '\t' : c == 'r' ? '\r' : c;
      }
    }
    if (pos_ >= s_.size()) return false;
    ++pos_;
    return true;
  }
  const std::string& s_;
  size_t pos_ = 0;
};

// Splits UTF-8 into code points (as strings).
std::vector<std::string> CodePoints(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    size_t n = 1;
    while (i + n < s.size() && (static_cast<unsigned char>(s[i + n]) & 0xC0) == 0x80) ++n;
    out.push_back(s.substr(i, n));
    i += n;
  }
  return out;
}

uint32_t Decode(const std::string& cp) {
  const auto* b = reinterpret_cast<const unsigned char*>(cp.data());
  if (cp.empty()) return 0;
  if (b[0] < 0x80) return b[0];
  int n = b[0] >= 0xF0 ? 3 : b[0] >= 0xE0 ? 2 : 1;
  uint32_t v = b[0] & (0x3F >> n);
  for (int k = 1; k <= n && k < static_cast<int>(cp.size()); ++k) v = (v << 6) | (b[k] & 0x3F);
  return v;
}

std::string ReplaceAll(std::string s, const std::string& from, const std::string& to) {
  for (size_t i = s.find(from); i != std::string::npos; i = s.find(from, i + to.size())) {
    s.replace(i, from.size(), to);
  }
  return s;
}

std::string AsciiLower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

std::vector<std::string> Split(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0, j; i <= s.size(); i = j + 1) {
    j = std::min(s.find(' ', i), s.size());
    if (j > i) out.push_back(s.substr(i, j - i));
  }
  return out;
}

// Script groups for the spacing rule (ICU script extensions, as far as
// Japanese text needs them).
enum : unsigned { kCjk = 1, kSoutheastAsian = 2 };
unsigned Scripts(uint32_t c) {
  if ((c >= 0x3400 && c <= 0x4DBF) || (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0xF900 && c <= 0xFAFF) ||
      (c >= 0x20000 && c <= 0x3FFFF) || c == 0x3005 || c == 0x3007 || (c >= 0x3021 && c <= 0x3029) ||
      (c >= 0x3038 && c <= 0x303B) || (c >= 0x2E80 && c <= 0x2FDF) || (c >= 0x3041 && c <= 0x30FF) ||
      (c >= 0x31F0 && c <= 0x31FF) || (c >= 0x32D0 && c <= 0x32FE) || (c >= 0xFF66 && c <= 0xFF9F) ||
      (c >= 0x3001 && c <= 0x3003) || (c >= 0x3008 && c <= 0x3011) || (c >= 0x3013 && c <= 0x301F) ||
      (c >= 0xFF61 && c <= 0xFF65) || (c >= 0x3190 && c <= 0x319F) || (c >= 0x3300 && c <= 0x33FF)) {
    return kCjk;
  }
  if ((c >= 0x0E00 && c <= 0x0EFF) || (c >= 0x1000 && c <= 0x109F) || (c >= 0x1780 && c <= 0x17FF)) {
    return kSoutheastAsian;
  }
  return 0;
}

bool IsDigit(uint32_t c) { return c >= '0' && c <= '9'; }

// Whether a space stays between two words of an unsegmented language.
bool NeedsSpace(const std::string& left, const std::string& right) {
  if (left.empty() || right.empty()) return false;
  const std::vector<std::string> lc = CodePoints(left), rc = CodePoints(right);
  const uint32_t l = Decode(lc.back()), r = Decode(rc.front());
  static const std::set<uint32_t> kNoSpaceAfter = {0x3001, 0x3002, 0x300A, 0x300B, 0x300C,
                                                   0x300D, 0xFF01, 0xFF08, 0xFF09, 0xFF0C,
                                                   0xFF1A, 0xFF1B, 0xFF1F, 0xFF5C, 0xFF5E};
  if (kNoSpaceAfter.count(l)) return false;
  const unsigned ls = Scripts(l), rs = Scripts(r);
  if (lc.size() >= 2 && lc[lc.size() - 2] == "°" && (l == 'C' || l == 'c') && (rs & kCjk)) return false;
  if (l == '\n' || l == ' ') return false;
  if ((IsDigit(l) && (rs & kCjk)) || (IsDigit(r) && (ls & kCjk))) return false;
  if (ls & rs) return false;
  return true;
}

}  // namespace

bool SpokenPunctuationNormalizer::Init(const std::string& converter_file, const std::string& symbol_file,
                                       const std::string& model_file, std::string* error) {
  if (!LoadConfig(converter_file, error)) return false;
  std::ifstream syms(symbol_file);
  if (!syms) {
    *error = "cannot read " + symbol_file;
    return false;
  }
  std::string line;
  while (std::getline(syms, line)) {
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) continue;
    vocab_[line.substr(0, tab)] = std::stoi(line.substr(tab + 1));
    input_dim_ = std::max(input_dim_, std::stoi(line.substr(tab + 1)) + 1);
  }
  if (!model_.Load(model_file, 1, error)) return false;
  const TfLiteTensor* in = model_.interpreter()->tensor(model_.interpreter()->inputs()[0]);
  if (!in->dims || in->dims->data[in->dims->size - 1] != input_dim_) {
    *error = "PUNCTUATION_LSTM input size differs from its symbol table";
    return false;
  }
  return true;
}

bool SpokenPunctuationNormalizer::LoadConfig(const std::string& path, std::string* error) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  TextMessage root;
  if (!f || !TextParser(text).Parse(&root)) {
    *error = "cannot parse " + path;
    return false;
  }
  for (const auto& [name, config] : root.messages) {
    if (name != "punctuation_config") continue;
    unsegmented_ = config.Get("unsegmented_language") == "true";
    for (const auto& [sname, s] : config.messages) {
      if (sname != "punctuation_symbols") continue;
      Symbol symbol;
      symbol.orthographic_form = s.Get("orthographic_form");
      symbol.textual_form = s.Get("textual_form");
      symbol.partner = s.Get("partner");
      const std::string side = s.Get("side");
      symbol.side = side == "LEFT" ? kLeft : side == "RIGHT" ? kRight : kNone;
      const int index = static_cast<int>(symbols_.size());
      symbols_.push_back(symbol);
      marks_.push_back(symbol.replacement());
      for (const auto& [fname, form] : s.messages) {
        if (fname != "spoken_forms") continue;
        const std::string words = form.Get("words");
        max_words_ = std::max<size_t>(max_words_, std::count(words.begin(), words.end(), ' ') + 1);
        Unsegmented u;
        u.key = form.Get("unsegmented_words");
        size_t bytes = 0;
        for (char c : words) {
          if (c == ' ') {
            u.cuts.push_back(bytes);
          } else {
            ++bytes;
          }
        }
        if (u.key.empty()) u.key = ReplaceAll(words, " ", "");
        if (!u.key.empty()) unsegmented_forms_.push_back(u);
        if (!words.empty() && form.Get("disabled") != "true") forms_[words].push_back(index);
      }
    }
  }
  if (symbols_.empty()) {
    *error = "no punctuation symbols in " + path;
    return false;
  }
  return true;
}

int SpokenPunctuationNormalizer::Find(const std::string& token) const {
  const auto it = vocab_.find(token);
  return it == vocab_.end() ? -1 : it->second;
}

std::string SpokenPunctuationNormalizer::Join(const std::vector<Word>& words) const {
  std::vector<std::string> parts;
  for (const Word& w : words) {
    if (w.text.find_first_not_of(' ') != std::string::npos) parts.push_back(w.text);
  }
  std::string text;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0 && (!unsegmented_ || NeedsSpace(parts[i - 1], parts[i]))) text += ' ';
    text += parts[i];
  }
  return text;
}

// Whether the text contains a spoken form at all.
bool SpokenPunctuationNormalizer::Gate(const std::string& text) const {
  for (size_t p = 0; p < text.size(); ++p) {
    if ((static_cast<unsigned char>(text[p]) & 0xC0) == 0x80) continue;
    for (const Unsegmented& u : unsegmented_forms_) {
      if (text.compare(p, u.key.size(), u.key) == 0) return true;
    }
  }
  return false;
}

// Marks word boundaries at the spoken forms in unsegmented text. As in
// libsoda, only forms of ASCII bytes can match.
std::string SpokenPunctuationNormalizer::Segment(const std::string& text) const {
  const std::string t = ReplaceAll(text, " ", "　");
  std::set<size_t> cuts;
  for (size_t p = 0; p < t.size(); ++p) {
    if ((static_cast<unsigned char>(t[p]) & 0xC0) == 0x80) continue;
    for (const Unsegmented& u : unsegmented_forms_) {
      if (std::any_of(u.key.begin(), u.key.end(), [](char c) { return static_cast<unsigned char>(c) >= 0x80; })) continue;
      if (t.compare(p, u.key.size(), u.key) != 0) continue;
      if (p != 0) cuts.insert(p);
      for (size_t b : u.cuts) cuts.insert(p + b);
      cuts.insert(p + u.key.size());
    }
  }
  std::string out;
  size_t last = 0;
  for (size_t c : cuts) {
    if (c <= last || c >= t.size()) continue;
    out += t.substr(last, c - last) + " ";
    last = c;
  }
  return out + t.substr(last);
}

void SpokenPunctuationNormalizer::Tokenize(const std::string& text, std::vector<std::string>* model,
                                           std::vector<std::string>* orig) const {
  constexpr int kFirstWord = 71;  // Ids below are ASCII marks and letters.
  for (const std::string& w : Split(unsegmented_ ? Segment(text) : text)) {
    if (!model->empty()) {
      model->push_back("⎵");
      orig->push_back("⎵");
    }
    const std::string lw = AsciiLower(w);
    if (Find(lw) >= kFirstWord) {
      model->push_back(lw);
      orig->push_back(w);
      continue;
    }
    std::vector<std::string> lc, oc;
    if (unsegmented_) {
      lc = CodePoints(lw);
      oc = CodePoints(w);
    } else {
      for (size_t i = 0; i < w.size(); ++i) {
        lc.push_back(lw.substr(i, 1));
        oc.push_back(w.substr(i, 1));
      }
    }
    for (size_t i = 0; i < lc.size(); ++i) {
      model->push_back(Find(lc[i]) >= 0 ? lc[i] : "⍰");
      orig->push_back(oc[i]);
    }
  }
}

bool SpokenPunctuationNormalizer::Flag(const std::vector<std::string>& tokens, std::vector<bool>* flags) {
  tflite::Interpreter* interp = model_.interpreter();
  const int in = interp->inputs()[0];
  const int n = static_cast<int>(tokens.size());
  // Time-major: [steps, batch 1, one-hot].
  if (interp->ResizeInputTensor(in, {n, 1, input_dim_}) != kTfLiteOk || interp->AllocateTensors() != kTfLiteOk) {
    return false;
  }
  // Each result starts from a zero LSTM state.
  if (interp->ResetVariableTensors() != kTfLiteOk) return false;
  float* x = interp->typed_tensor<float>(in);
  std::fill(x, x + n * input_dim_, 0.0f);
  for (int t = 0; t < n; ++t) {
    const int id = Find(tokens[t]);
    if (id >= 0) x[t * input_dim_ + id] = 1.0f;
  }
  if (interp->Invoke() != kTfLiteOk) return false;
  const float* y = interp->typed_tensor<float>(interp->outputs()[0]);
  flags->resize(n);
  for (int t = 0; t < n; ++t) (*flags)[t] = y[2 * t + 1] > y[2 * t];
  return true;
}

// The longest spoken form that `text` starts with, ending at a word boundary.
std::string SpokenPunctuationNormalizer::LongestForm(const std::string& text) const {
  const std::string padded = " " + text + " ";
  size_t best = 0;
  for (const auto& [words, unused] : forms_) {
    if (words.size() > best && padded.compare(0, words.size() + 2, " " + words + " ") == 0) best = words.size();
  }
  return text.substr(0, best);
}

bool SpokenPunctuationNormalizer::Apply(std::vector<Word>* words) {
  if (words->empty()) return false;
  const std::string text = Join(*words);
  if (!Gate(text)) return false;
  std::vector<std::string> model, orig;
  Tokenize(text, &model, &orig);
  std::vector<bool> flags;
  if (model.empty() || !Flag(model, &flags)) return false;

  const auto is_mark = [&](const std::string& t) {
    return std::find(marks_.begin(), marks_.end(), t) != marks_.end();
  };
  const std::string kSep = "⎵";
  std::vector<std::pair<std::string, std::string>> spans;  // Spoken text, mark.
  std::map<std::string, int> open;                         // Open LEFT marks.
  std::string out;
  const size_t n = model.size(), limit = 2 * max_words_ + 2;
  for (size_t i = 0; i < n;) {
    if (!flags[i]) {
      if (is_mark(orig[i])) spans.emplace_back(orig[i], orig[i]);
      out += orig[i++];
      continue;
    }
    size_t j = i;
    while (j < n && j < i + limit && (flags[j] || orig[j] == kSep)) ++j;
    std::string m, o;
    for (size_t k = i; k < j; ++k) {
      m += model[k] == kSep ? " " : model[k];
      o += orig[k] == kSep ? " " : orig[k];
    }
    m = LongestForm(m);
    o = LongestForm(o);
    if (m.empty()) {
      if (is_mark(orig[i])) spans.emplace_back(orig[i], orig[i]);
      out += orig[i++];
      continue;
    }
    const size_t consumed = 2 * std::count(m.begin(), m.end(), ' ') + 1;
    const std::vector<int>& candidates = forms_.at(m);
    int chosen = candidates[0];
    if (candidates.size() > 1) {
      int left = -1, right = -1;
      chosen = -1;
      for (int c : candidates) {
        if (symbols_[c].side == kNone) {
          chosen = c;
          break;
        }
        (symbols_[c].side == kLeft ? left : right) = c;
      }
      if (chosen < 0) {
        chosen = right < 0 || (left >= 0 && open[symbols_[left].orthographic_form] <= 0) ? left : right;
      }
    }
    const Symbol& s = symbols_[chosen];
    if (s.side == kRight) {
      int& count = open[s.partner];
      count = std::max(0, count - 1);
    } else {
      ++open[s.orthographic_form];
    }
    if (unsegmented_) o = ReplaceAll(ReplaceAll(o, " ", ""), "　", " ");
    spans.emplace_back(o, s.replacement());
    out += s.replacement();
    i += consumed;
  }
  if (spans.empty()) return false;
  out = ReplaceAll(out, kSep, " ");
  if (unsegmented_) out = ReplaceAll(ReplaceAll(out, " ", ""), "　", " ");

  const std::vector<std::string> new_words = Split(out);
  if (new_words.empty() || new_words.size() > words->size()) return false;

  // Aligns the new text with the old words by character: unchanged words keep
  // their times, and each mark becomes a word spanning its spoken characters.
  const std::vector<Word>& old = *words;
  std::vector<std::vector<std::string>> ow;
  for (const Word& w : old) ow.push_back(CodePoints(ReplaceAll(w.text, " ", "")));
  std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> sp;
  for (const auto& [first, second] : spans) sp.emplace_back(CodePoints(ReplaceAll(first, " ", "")), CodePoints(second));
  std::vector<Word> result;
  Word cur{"", old[0].start_ms, old[0].start_ms};
  size_t j = 0, q = 0, k = 0, r = 0;
  bool ok = true;
  const auto advance = [&]() {
    if (++q == ow[j].size()) {
      cur.end_ms = old[j].end_ms;
      return true;
    }
    cur.end_ms = old[j].start_ms + (old[j].end_ms - old[j].start_ms) / static_cast<int64_t>(ow[j].size()) * q;
    return false;
  };
  for (const std::string& c : CodePoints(ReplaceAll(out, " ", ""))) {
    bool normal = true;
    if (k < sp.size() && r < sp[k].second.size() && c == sp[k].second[r]) {
      if (++r < sp[k].second.size()) continue;
      if (sp[k].first == sp[k].second) {
        ++k;
        r = 0;
      } else {
        normal = false;
        if (!cur.text.empty()) {
          result.push_back(cur);
          cur.text.clear();
          cur.start_ms = cur.end_ms;
        }
        for (const std::string& ch : sp[k].first) {
          if (j >= ow.size() || ch != ow[j][q]) {
            ok = false;
            break;
          }
          if (advance()) {
            ++j;
            q = 0;
          }
        }
        if (!ok) break;
        for (const std::string& ch : sp[k].second) cur.text += ch;
        result.push_back(cur);
        cur.text.clear();
        cur.start_ms = cur.end_ms;
        if (q == 0 && j < old.size()) cur.start_ms = old[j].start_ms;
        ++k;
        r = 0;
      }
    }
    if (!normal) continue;
    if (j >= ow.size() || c != ow[j][q]) {
      ok = false;
      break;
    }
    cur.text += c;
    if (advance()) {
      result.push_back(cur);
      ++j;
      q = 0;
      if (j < old.size()) cur = {"", old[j].start_ms, old[j].start_ms};
    }
  }
  if (!ok) {
    // libsoda then spreads the new words evenly over the old words' time.
    result.clear();
    const int64_t s = old.front().start_ms, e = old.back().end_ms;
    const int64_t d = (e - s) / static_cast<int64_t>(new_words.size());
    for (size_t i = 0; i < new_words.size(); ++i) {
      result.push_back({new_words[i], s + static_cast<int64_t>(i) * d,
                        i + 1 == new_words.size() ? e : s + static_cast<int64_t>(i + 1) * d});
    }
  }
  *words = std::move(result);
  return true;
}

}  // namespace soda

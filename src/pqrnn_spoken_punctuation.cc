#include "pqrnn_spoken_punctuation.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>

#include "proto_wire.h"
#include "punctuator.h"
#include "unicode_case.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

namespace soda {
namespace {

// speech_formatting.TokenSegment.BreakLevel: what comes before a token.
enum Break { kUnset = -1, kEmpty = 0, kSpace = 1, kLine = 2, kParagraph = 3, kCustom = 4 };
// BasicTextSegmenterConfig.SegmentBy.
constexpr int kUnicodeCharacter = 2, kWhitespaceAndPunctuation = 3;

bool OneChar(const std::string& s) { return !s.empty() && CharLength(s, 0) == s.size(); }

void ReplaceAll(std::string* s, const std::string& from, const std::string& to) {
  if (from.empty()) return;
  for (size_t i = s->find(from); i != std::string::npos; i = s->find(from, i + to.size())) {
    s->replace(i, from.size(), to);
  }
}

}  // namespace

struct PqrnnSpokenPunctuationNormalizer::Token {
  int break_level = kUnset;
  std::string custom_break;
  std::string token, prepend, append;
  bool end_of_sentence = false;
  // The input words it covers (libsoda: frames), or -1.
  int first = -1, last = -1;

  std::string BreakText() const {
    switch (break_level) {
      case kSpace: return " ";
      case kLine: return "\n";
      case kParagraph: return "\n\n";
      case kCustom: return custom_break;
      default: return "";
    }
  }
  std::string Text() const { return BreakText() + prepend + token + append; }
};

bool PqrnnSpokenPunctuationNormalizer::Init(const std::string& model_file, std::string_view settings,
                                            std::string* error) {
  ProtoReader r(settings);
  int segment_by = 0;
  while (r.Next()) {
    switch (r.field()) {
      case 2: capitalization_ = static_cast<int>(r.int64()); break;
      case 4: {  // postnormalizers: InferenceNormalizer {1 pattern, 2 replace_with, 3 type}.
        Replacement p;
        ProtoReader n(r.bytes());
        while (n.Next()) {
          if (n.field() == 1) p.from = n.str();
          if (n.field() == 2) p.to = n.str();
          if (n.field() == 3) p.type = static_cast<int>(n.int64());
        }
        postnormalizers_.push_back(p);
        break;
      }
      case 5: {  // text_segmenter_config: BasicTextSegmenterConfig.
        ProtoReader seg(r.bytes());
        while (seg.Next()) {
          if (seg.field() == 1) marks_.push_back(seg.str());
          if (seg.field() == 2) sentence_end_marks_.push_back(seg.str());
          if (seg.field() == 3) segment_by = static_cast<int>(seg.int64());
        }
        break;
      }
      case 6: {  // supported_punctuation: PunctuationSymbol.
        Symbol s;
        std::vector<std::string> forms;
        ProtoReader p(r.bytes());
        while (p.Next()) {
          switch (p.field()) {
            case 1: s.mark = p.str(); break;
            case 3: forms.push_back(p.str()); break;
            case 4: s.separate_by_space = p.boolean(); break;
            case 5: s.type = static_cast<int>(p.int64()); break;
            case 6: s.partner = p.str(); break;
            case 7: s.side = static_cast<int>(p.int64()); break;
            case 8: s.end_of_sentence = p.boolean(); break;
          }
        }
        for (const auto& f : forms) forms_[f].push_back(static_cast<int>(symbols_.size()));
        symbols_.push_back(s);
        break;
      }
      case 7: separate_words_by_space_ = r.boolean(); break;
    }
  }
  by_whitespace_ = segment_by != kUnicodeCharacter;
  if (segment_by != kWhitespaceAndPunctuation) marks_.clear();
  // Longest marks first, so "..." is not read as three ".".
  std::stable_sort(marks_.begin(), marks_.end(),
                   [](const std::string& a, const std::string& b) { return a.size() > b.size(); });

  model_ = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
  if (!model_) {
    *error = "cannot read " + model_file;
    return false;
  }
  tflite::ops::builtin::BuiltinOpResolver resolver;
  AddPqrnnOps(&resolver);
  if (tflite::InterpreterBuilder(*model_, resolver)(&interpreter_) != kTfLiteOk || !interpreter_ ||
      interpreter_->AllocateTensors() != kTfLiteOk) {
    *error = "cannot build " + model_file;
    return false;
  }
  for (int idx : interpreter_->outputs()) {
    const char* name = interpreter_->tensor(idx)->name;
    if (const char* slash = std::strrchr(name, '/')) name = slash + 1;
    if (!std::strcmp(name, "Softmax")) scores_output_ = idx;
    if (!std::strcmp(name, "LabelMap")) labels_output_ = idx;
  }
  if (scores_output_ < 0 || labels_output_ < 0 || symbols_.empty()) {
    *error = model_file + ": unsupported spoken punctuation model";
    return false;
  }
  return true;
}

// BasicTextSegmenter: whitespace sets the next piece's break, and a mark
// splits the word, joining the piece before it (or, at the start, the piece
// after it).
std::vector<PqrnnSpokenPunctuationNormalizer::Token> PqrnnSpokenPunctuationNormalizer::Segment(
    const std::string& word, int index) const {
  std::vector<Token> out;
  Token cur;
  cur.break_level = kEmpty;
  cur.first = cur.last = index;
  auto push = [&](int next_break) {
    if (!cur.token.empty() || !cur.prepend.empty() || !cur.append.empty()) {
      // A word of marks only keeps them as its punctuation.
      if (cur.token.empty() && cur.append.empty()) std::swap(cur.prepend, cur.append);
      out.push_back(cur);
    }
    cur = Token();
    cur.break_level = next_break;
    cur.first = cur.last = index;
  };
  for (size_t i = 0; i < word.size();) {
    if (std::strchr(" \t\n\r", word[i])) {
      size_t j = i;
      while (j < word.size() && std::strchr(" \t\n\r", word[j])) ++j;
      const std::string run = word.substr(i, j - i);
      push(run == " " ? kSpace : run == "\n" ? kLine : run == "\n\n" ? kParagraph : kCustom);
      cur.custom_break = run;
      i = j;
      continue;
    }
    const std::string* mark = nullptr;
    for (const auto& m : marks_) {
      if (word.compare(i, m.size(), m) == 0) {
        mark = &m;
        break;
      }
    }
    if (mark) {
      if (cur.token.empty() && cur.append.empty()) {
        cur.prepend += *mark;
      } else {
        cur.append += *mark;
        cur.end_of_sentence |= std::find(sentence_end_marks_.begin(), sentence_end_marks_.end(), *mark) !=
                               sentence_end_marks_.end();
      }
      i += mark->size();
      continue;
    }
    if (!cur.append.empty()) push(kEmpty);
    const size_t n = CharLength(word, i);
    cur.token += word.substr(i, n);
    i += n;
  }
  push(kEmpty);
  return out;
}

bool PqrnnSpokenPunctuationNormalizer::Label(const std::vector<Token>& tokens,
                                             std::vector<bool>* punctuation) const {
  std::string text;
  size_t count = 0;
  for (const auto& t : tokens) {
    if (t.token.find_first_not_of(" \t\n\r\f\v") == std::string::npos) continue;
    if (count++ > 0) text += ' ';
    text += t.token;
  }
  punctuation->clear();
  if (count == 0) return true;
  tflite::DynamicBuffer buf;
  buf.AddString(text.data(), text.size());
  buf.WriteToTensorAsVector(interpreter_->tensor(interpreter_->inputs()[0]));
  if (interpreter_->Invoke() != kTfLiteOk) return false;
  const TfLiteTensor* lt = interpreter_->tensor(labels_output_);
  std::vector<std::string> labels;
  for (int i = 0, n = tflite::GetStringCount(lt); i < n; ++i) {
    const tflite::StringRef s = tflite::GetString(lt, i);
    labels.emplace_back(s.str, s.len);
  }
  const TfLiteTensor* st = interpreter_->tensor(scores_output_);
  const int classes = st->dims->data[st->dims->size - 1];
  int rows = 1;
  for (int d = 0; d + 1 < st->dims->size; ++d) rows *= st->dims->data[d];
  if (rows != static_cast<int>(count) || classes != static_cast<int>(labels.size())) return false;
  auto score = [&](int i) {
    return st->type == kTfLiteUInt8 ? (st->data.uint8[i] - st->params.zero_point) * st->params.scale
                                    : st->data.f[i];
  };
  for (int r = 0; r < rows; ++r) {
    int best = 0;  // Ties go to the first label.
    for (int c = 1; c < classes; ++c) {
      if (score(r * classes + c) > score(r * classes + best)) best = c;
    }
    punctuation->push_back(labels[best] != "KEEP");
  }
  return true;
}

// speech_formatting's WordFormatter, one per request.
std::vector<PqrnnSpokenPunctuationNormalizer::Token> PqrnnSpokenPunctuationNormalizer::Format(
    const std::vector<Token>& tokens, const std::vector<bool>& punctuation) const {
  std::vector<Token> out, pending;
  std::map<std::string, int> open;  // Opened marks of pairs.
  bool sentence_ended = false;
  int previous_type = kUndefined;
  auto key = [&] {
    std::string k;
    for (const auto& t : pending) k += (k.empty() || !by_whitespace_ ? "" : " ") + t.token;
    return Lower(k);
  };
  auto choose = [&](const std::vector<int>& candidates) {
    int chosen = candidates.front();
    for (int i : candidates) {
      const Symbol& s = symbols_[i];
      if (s.partner.empty()) continue;
      if (s.side == kRight) {
        auto [it, inserted] = open.try_emplace(s.partner, 0);
        if (!inserted && it->second > 0) chosen = i;
      } else if (s.side == kLeft) {
        auto [it, inserted] = open.try_emplace(s.mark, 0);
        if (inserted || it->second == 0) chosen = i;
      }
    }
    const Symbol& c = symbols_[chosen];
    if (c.side == kLeft) ++open[c.mark];
    if (c.side == kRight) {
      auto [it, inserted] = open.try_emplace(c.partner, 0);
      if (it->second >= 1) --it->second;
    }
    return chosen;
  };
  std::function<void()> flush;
  std::function<void(const Token&)> add_plain;
  flush = [&] {
    if (pending.empty()) return;
    auto form = forms_.find(key());
    if (form == forms_.end()) {  // Not a whole spoken form: plain words.
      std::vector<Token> words;
      words.swap(pending);
      for (const auto& t : words) add_plain(t);
      return;
    }
    const Symbol& s = symbols_[choose(form->second)];
    // Where the mark goes: a new token at the start or after a line break,
    // else the last one, as its type allows.
    bool fresh = out.empty() || pending.front().break_level >= kLine;
    if (!fresh) {
      const Token& last = out.back();
      switch (s.type) {
        case kFinal: break;
        case kInitial: fresh = !last.token.empty() || !last.append.empty(); break;
        case kIntermediate: fresh = !last.append.empty(); break;
        default: fresh = true;
      }
    }
    if (fresh) out.emplace_back();
    Token& dst = out.back();
    if (dst.first < 0) dst.first = pending.front().first;
    dst.last = pending.back().last;
    if (dst.break_level == kUnset) {
      dst.break_level = pending.front().break_level;
      dst.custom_break = pending.front().custom_break;
    }
    switch (s.type) {
      case kInitial: dst.prepend += s.separate_by_space ? s.mark + " " : s.mark; break;
      case kIntermediate: dst.token += s.separate_by_space ? " " + s.mark + " " : s.mark; break;
      case kFinal:
        dst.append += s.separate_by_space ? " " + s.mark : s.mark;
        dst.end_of_sentence |= s.end_of_sentence;
        break;
      default:
        dst.token += s.mark;
        dst.end_of_sentence |= s.end_of_sentence;
    }
    sentence_ended |= s.end_of_sentence;
    pending.clear();
    previous_type = s.type;
  };
  add_plain = [&](const Token& tok) {
    flush();
    Token* last = out.empty() ? nullptr : &out.back();
    if (!last) previous_type = kUndefined;
    if (tok.break_level >= kLine) previous_type = kStandalone;
    auto extend = [&] {  // The word joins the last token.
      last->append = tok.append;
      last->last = tok.last;
      last->end_of_sentence = tok.end_of_sentence;
      previous_type = kUndefined;
    };
    if (previous_type == kInitial) {  // Into the opening mark's placeholder.
      last->prepend += tok.prepend;
      last->token = tok.token;
      return extend();
    }
    if (previous_type == kIntermediate && tok.prepend.empty()) {
      last->token += tok.token;
      return extend();
    }
    if ((previous_type == kUndefined || previous_type == kFinal) && last && tok.prepend.empty() &&
        tok.token.empty() && OneChar(tok.append)) {  // A word that is one mark.
      last->append += tok.append;
      last->last = tok.last;
      last->end_of_sentence = tok.end_of_sentence;
      previous_type = kUndefined;
      return;
    }
    Token word = tok;
    if (last && last->end_of_sentence) {
      if (word.break_level != kEmpty || !by_whitespace_ ||
          (last->last >= 0 && word.first >= 0 && last->last < word.first)) {
        if (capitalization_ != 0) Capitalize(&word.token);
      } else {
        last->end_of_sentence = false;
      }
    }
    out.push_back(word);
    previous_type = kUndefined;
  };

  for (size_t i = 0, k = 0; i < tokens.size(); ++i) {
    const Token& tok = tokens[i];
    if (tok.token.find_first_not_of(" \t\n\r\f\v") == std::string::npos) {
      add_plain(tok);
      continue;
    }
    if (!punctuation[k++]) {
      flush();
      add_plain(tok);
      continue;
    }
    pending.push_back(tok);
    const std::string k_text = key();
    auto it = forms_.lower_bound(k_text);
    if (it == forms_.end() || it->first.compare(0, k_text.size(), k_text) != 0) {
      pending.pop_back();
      flush();
      pending.push_back(tok);
    }
  }
  flush();
  // CAPITALIZE_ALL (2), or CAPITALIZE_AFTER_EOS_AND_FIRST_WORD_PRECEDING_EOS
  // (3) once a spoken mark ended a sentence.
  if (!out.empty() && (capitalization_ == 2 || (capitalization_ == 3 && sentence_ended))) {
    Capitalize(&out.front().token);
  }
  return out;
}

std::vector<PqrnnSpokenPunctuationNormalizer::Token> PqrnnSpokenPunctuationNormalizer::Run(
    const std::vector<std::string>& words) const {
  // The words, each split at its punctuation; a word's first piece follows a
  // space.
  std::vector<Token> tokens;
  for (size_t index = 0; index < words.size(); ++index) {
    std::vector<Token> pieces = Segment(words[index], static_cast<int>(index));
    if (pieces.empty()) {
      pieces.emplace_back();
      pieces.back().break_level = kEmpty;
      pieces.back().first = pieces.back().last = static_cast<int>(index);
    }
    if (separate_words_by_space_ && !tokens.empty() && pieces.front().break_level == kEmpty) {
      const std::string before = tokens.back().Text();
      if (before.find_first_not_of(" \t\n\r") != std::string::npos) pieces.front().break_level = kSpace;
    }
    tokens.insert(tokens.end(), pieces.begin(), pieces.end());
  }
  if (tokens.empty()) return {};
  std::vector<bool> punctuation;
  if (!Label(tokens, &punctuation)) return {};
  const std::vector<Token> formatted = Format(tokens, punctuation);

  // The pieces of one word join again.
  std::vector<Token> merged;
  for (const Token& t : formatted) {
    if (!merged.empty()) {
      Token& prev = merged.back();
      if (prev.first >= 0 && prev.last >= 0 && t.first >= 0 && prev.last >= t.first) {
        prev.token += prev.append + t.BreakText() + t.prepend + t.token;
        prev.append = t.append;
        prev.last = t.last;
        prev.end_of_sentence = t.end_of_sentence;
        continue;
      }
    }
    merged.push_back(t);
  }
  // InferenceNormalizer types: UNPUNCTUATED_WORD (2), APPENDED_PUNCTUATION
  // (3), PREPENDED_PUNCTUATION (4).
  for (Token& t : merged) {
    for (const auto& p : postnormalizers_) {
      if (p.type == 2) ReplaceAll(&t.token, p.from, p.to);
      if (p.type == 3) ReplaceAll(&t.append, p.from, p.to);
      if (p.type == 4) ReplaceAll(&t.prepend, p.from, p.to);
    }
  }
  return merged;
}

std::string PqrnnSpokenPunctuationNormalizer::Apply(const std::string& text) const {
  std::vector<std::string> words;
  for (size_t i = 0, j; i <= text.size(); i = j + 1) {
    j = std::min(text.find(' ', i), text.size());
    if (j > i) words.push_back(text.substr(i, j - i));
  }
  const std::vector<Token> tokens = Run(words);
  if (tokens.empty()) return text;
  std::string out;
  for (const Token& t : tokens) out += t.Text();
  return out;
}

bool PqrnnSpokenPunctuationNormalizer::ApplyWords(std::vector<Word>* words) const {
  std::vector<std::string> texts;
  for (const Word& w : *words) texts.push_back(w.text);
  const std::vector<Token> tokens = Run(texts);
  if (tokens.empty()) return false;
  // ToDecodedWords: a token's frames are those of the words it covers, and
  // its text leaves out the space before it but not a line break.
  std::vector<Word> out;
  for (const Token& t : tokens) {
    Word w;
    w.text = (t.break_level >= kLine ? t.BreakText() : "") + t.prepend + t.token + t.append;
    if (t.first >= 0) {
      w.start_ms = (*words)[t.first].start_ms;
      w.end_ms = (*words)[std::max(t.first, t.last)].end_ms;
    } else if (!out.empty()) {
      w.start_ms = w.end_ms = out.back().end_ms;
    }
    out.push_back(std::move(w));
  }
  *words = std::move(out);
  return true;
}

}  // namespace soda

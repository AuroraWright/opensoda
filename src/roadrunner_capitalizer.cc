#include "roadrunner_capitalizer.h"

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <utility>

#include "flatbuffers/flexbuffers.h"
#include "proto_wire.h"
#include "punctuator.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"

namespace soda {
namespace {

// ---- speech.textnorm.Sentence, the fields this model uses ----

struct Example {
  std::vector<std::string> input;       // 1 input_token_pieces
  std::vector<std::string> hypothesis;  // 5 hypothesis_token_pieces
  bool skip_in_training = false;        // 10
  bool has_span = false;
  int span_left = 0, span_right = 0;  // 12 context_span
  bool skip_in_inference = false;     // 13
};

struct Sentence {
  std::vector<Example> examples;  // 1
  bool has_text = false;
  std::string original_text;                     // 2
  std::vector<std::string> context;              // 3 context_token_pieces
  std::vector<std::pair<int, int>> spans;        // 4, byte offsets
  bool has_score = false;
  float score = 0;            // 6
  std::string input_request;  // 8, passed through as is
};

std::pair<int, int> ParseSpan(std::string_view b) {
  std::pair<int, int> s{0, 0};
  for (ProtoReader r(b); r.Next();) {
    if (r.field() == 1) s.first = static_cast<int>(r.int64());
    if (r.field() == 2) s.second = static_cast<int>(r.int64());
  }
  return s;
}

Sentence Parse(std::string_view data) {
  Sentence s;
  for (ProtoReader r(data); r.Next();) {
    switch (r.field()) {
      case 1: {
        Example e;
        for (ProtoReader x(r.bytes()); x.Next();) {
          switch (x.field()) {
            case 1: e.input.push_back(x.str()); break;
            case 5: e.hypothesis.push_back(x.str()); break;
            case 10: e.skip_in_training = x.boolean(); break;
            case 12:
              e.has_span = true;
              std::tie(e.span_left, e.span_right) = ParseSpan(x.bytes());
              break;
            case 13: e.skip_in_inference = x.boolean(); break;
          }
        }
        s.examples.push_back(std::move(e));
        break;
      }
      case 2:
        s.has_text = true;
        s.original_text = r.str();
        break;
      case 3: s.context.push_back(r.str()); break;
      case 4: s.spans.push_back(ParseSpan(r.bytes())); break;
      case 6:
        s.has_score = true;
        s.score = r.fixed_float();
        break;
      case 8: s.input_request = r.str(); break;
    }
  }
  return s;
}

void WriteSpan(ProtoWriter* w, int field, int left, int right) {
  ProtoWriter span;
  span.Varint(1, left);
  span.Varint(2, right);
  w->Message(field, span);
}

std::string Serialize(const Sentence& s) {
  ProtoWriter w;
  for (const Example& e : s.examples) {
    ProtoWriter x;
    for (const auto& p : e.input) x.Bytes(1, p);
    for (const auto& p : e.hypothesis) x.Bytes(5, p);
    if (e.skip_in_training) x.Bool(10, true);
    if (e.has_span) WriteSpan(&x, 12, e.span_left, e.span_right);
    if (e.skip_in_inference) x.Bool(13, true);
    w.Message(1, x);
  }
  if (s.has_text) w.Bytes(2, s.original_text);
  for (const auto& c : s.context) w.Bytes(3, c);
  for (const auto& [left, right] : s.spans) WriteSpan(&w, 4, left, right);
  if (s.has_score) w.Float(6, s.score);
  if (!s.input_request.empty()) w.Bytes(8, s.input_request);
  return w.data();
}

// UTF-8 characters of `s`.
std::vector<std::string> Characters(const std::string& s) {
  std::vector<std::string> out;
  for (size_t i = 0; i < s.size();) {
    const auto c = static_cast<unsigned char>(s[i]);
    size_t n = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
    n = std::min(n, s.size() - i);
    out.push_back(s.substr(i, n));
    i += n;
  }
  return out;
}

// ---- tensor helpers ----

TfLiteTensor* Input(TfLiteContext* context, TfLiteNode* node, int i) {
  return &context->tensors[node->inputs->data[i]];
}
TfLiteTensor* Output(TfLiteContext* context, TfLiteNode* node, int i) {
  return &context->tensors[node->outputs->data[i]];
}

std::vector<std::string> Strings(const TfLiteTensor* t) {
  std::vector<std::string> out;
  for (int i = 0, n = tflite::GetStringCount(t); i < n; ++i) {
    const tflite::StringRef s = tflite::GetString(t, i);
    out.emplace_back(s.str, s.len);
  }
  return out;
}

TfLiteIntArray* Dims(std::initializer_list<int> d) {
  TfLiteIntArray* a = TfLiteIntArrayCreate(static_cast<int>(d.size()));
  std::copy(d.begin(), d.end(), a->data);
  return a;
}

void WriteStrings(const std::vector<std::string>& v, TfLiteTensor* t, TfLiteIntArray* dims) {
  tflite::DynamicBuffer buf;
  for (const auto& s : v) buf.AddString(s.data(), s.size());
  buf.WriteToTensor(t, dims);
}

TfLiteStatus WriteInts(TfLiteContext* context, const std::vector<int32_t>& v, TfLiteTensor* t,
                       TfLiteIntArray* dims) {
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, t, dims));
  std::copy(v.begin(), v.end(), t->data.i32);
  return kTfLiteOk;
}

TfLiteStatus DynamicOutputs(TfLiteContext* context, TfLiteNode* node) {
  for (int i = 0; i < node->outputs->size; ++i) Output(context, node, i)->allocation_type = kTfLiteDynamic;
  return kTfLiteOk;
}

std::string StringOption(const char* buffer, size_t length, const char* key) {
  const auto m = flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length).AsMap();
  return m[key].AsString().str();
}

// Padding and space replacement of the ops that turn words into tensors.
struct TokenOptions {
  std::string padding = "<epsilon>";
  std::string replace_space = "<spc>";
  std::string Token(const std::string& s) const {
    std::string out;
    for (char c : s) {
      if (c == ' ') {
        out += replace_space;
      } else {
        out += c;
      }
    }
    return out;
  }
};

void* TokenOptionsInit(TfLiteContext*, const char* buffer, size_t length) {
  auto* o = new TokenOptions;
  o->padding = StringOption(buffer, length, "padding");
  o->replace_space = StringOption(buffer, length, "replace_space");
  return o;
}
void TokenOptionsFree(TfLiteContext*, void* p) { delete static_cast<TokenOptions*>(p); }

// ---- SentenceToInputContext: the words of each sentence, padded ----

TfLiteStatus InputContextEval(TfLiteContext* context, TfLiteNode* node) {
  const auto& o = *static_cast<TokenOptions*>(node->user_data);
  const std::vector<std::string> in = Strings(Input(context, node, 0));
  std::vector<Sentence> sentences;
  size_t most = 0;
  for (const auto& s : in) {
    sentences.push_back(Parse(s));
    most = std::max(most, sentences.back().context.size());
  }
  std::vector<std::string> words;
  std::vector<int32_t> counts;
  for (const Sentence& s : sentences) {
    for (size_t i = 0; i < most; ++i) words.push_back(i < s.context.size() ? o.Token(s.context[i]) : o.padding);
    counts.push_back(static_cast<int32_t>(s.context.size()));
  }
  const int b = static_cast<int>(sentences.size());
  WriteStrings(words, Output(context, node, 0), Dims({b, static_cast<int>(most)}));
  return WriteInts(context, counts, Output(context, node, 1), Dims({b}));
}

// ---- RetokenizeSentence: one example per word, from the tagger's labels ----

// The model's tag set has one hypothesis class, <self>: label 1 keeps a word
// as it is, label 2 has the seq2seq model rewrite it.
constexpr int kKeepLabel = 1, kRewriteLabel = 2;

TfLiteStatus RetokenizeEval(TfLiteContext* context, TfLiteNode* node) {
  const std::vector<std::string> in = Strings(Input(context, node, 0));
  const TfLiteTensor* labels = Input(context, node, 1);
  const TfLiteTensor* counts = Input(context, node, 2);
  const TfLiteTensor* scores = Input(context, node, 3);
  TF_LITE_ENSURE(context, labels->dims->size == 3);
  const int tokenizations = labels->dims->data[1], width = labels->dims->data[2];
  std::vector<std::string> out;
  for (size_t b = 0; b < in.size(); ++b) {
    Sentence s = Parse(in[b]);
    s.input_request.clear();
    const int n = std::min<int>(counts->data.i32[b], static_cast<int>(s.context.size()));
    for (int i = 0; i < n; ++i) {
      const size_t at = b * tokenizations * width + i;
      const int label = labels->data.i32[at];
      if (label != kKeepLabel && label != kRewriteLabel) continue;
      Example e;
      e.input = Characters(s.context[i]);
      if (label == kKeepLabel) {
        e.hypothesis = {"<self>"};
        e.skip_in_training = e.skip_in_inference = true;
      }
      e.has_span = true;
      e.span_left = i;
      e.span_right = i + 1;
      s.examples.push_back(std::move(e));
      s.has_score = true;
      s.score += scores->data.f[at];
    }
    out.push_back(Serialize(s));
  }
  WriteStrings(out, Output(context, node, 0), Dims({static_cast<int>(out.size())}));
  return kTfLiteOk;
}

// ---- SentenceToTokenTensors: the words to rewrite, as character tensors ----

TfLiteStatus TokenTensorsEval(TfLiteContext* context, TfLiteNode* node) {
  const auto& o = *static_cast<TokenOptions*>(node->user_data);
  const std::vector<std::string> in = Strings(Input(context, node, 0));
  std::vector<int32_t> batch, tokenization, example, spans, lengths, target_lengths;
  std::vector<std::vector<std::string>> chars;
  std::vector<std::string> joined;
  size_t longest = 1;
  for (size_t b = 0; b < in.size(); ++b) {
    const Sentence s = Parse(in[b]);
    for (size_t i = 0; i < s.examples.size(); ++i) {
      const Example& e = s.examples[i];
      if (e.skip_in_inference) continue;
      batch.push_back(static_cast<int32_t>(b));
      tokenization.push_back(0);
      example.push_back(static_cast<int32_t>(i));
      spans.push_back(e.span_left);
      spans.push_back(e.span_right);
      chars.emplace_back();
      std::string word;
      for (const auto& p : e.input) {
        chars.back().push_back(o.Token(p));
        word += p;
      }
      lengths.push_back(static_cast<int32_t>(e.input.size()));
      target_lengths.push_back(0);
      joined.push_back(word);
      longest = std::max(longest, e.input.size());
    }
  }
  const int n = static_cast<int>(batch.size());
  std::vector<std::string> padded;
  for (auto& c : chars) {
    c.resize(longest, o.padding);
    padded.insert(padded.end(), c.begin(), c.end());
  }
  TF_LITE_ENSURE_OK(context, WriteInts(context, batch, Output(context, node, 0), Dims({n})));
  TF_LITE_ENSURE_OK(context, WriteInts(context, tokenization, Output(context, node, 1), Dims({n})));
  TF_LITE_ENSURE_OK(context, WriteInts(context, example, Output(context, node, 2), Dims({n})));
  TF_LITE_ENSURE_OK(context, WriteInts(context, spans, Output(context, node, 3), Dims({n, 2})));
  WriteStrings(padded, Output(context, node, 4), Dims({n, static_cast<int>(longest)}));
  TF_LITE_ENSURE_OK(context, WriteInts(context, lengths, Output(context, node, 5), Dims({n})));
  // No targets at inference.
  WriteStrings(std::vector<std::string>(n, o.padding), Output(context, node, 6), Dims({n, 1}));
  TF_LITE_ENSURE_OK(context, WriteInts(context, target_lengths, Output(context, node, 7), Dims({n})));
  WriteStrings(joined, Output(context, node, 8), Dims({n}));
  return kTfLiteOk;
}

// ---- AttachHypothesisToSentence: the rewritten words back into examples ----

TfLiteStatus AttachHypothesisEval(TfLiteContext* context, TfLiteNode* node) {
  const std::vector<std::string> in = Strings(Input(context, node, 0));
  const TfLiteTensor* batch = Input(context, node, 1);
  const TfLiteTensor* example = Input(context, node, 3);
  const TfLiteTensor* hyps = Input(context, node, 4);
  const TfLiteTensor* lengths = Input(context, node, 5);
  const TfLiteTensor* scores = Input(context, node, 6);
  std::vector<Sentence> sentences;
  for (const auto& s : in) sentences.push_back(Parse(s));
  const std::vector<std::string> pieces = Strings(hyps);
  const int n = batch->dims->data[0];
  const int width = hyps->dims->size == 2 ? hyps->dims->data[1] : 0;
  for (int i = 0; i < n; ++i) {
    const int b = batch->data.i32[i], x = example->data.i32[i];
    TF_LITE_ENSURE(context, b >= 0 && b < static_cast<int>(sentences.size()));
    Sentence& s = sentences[b];
    TF_LITE_ENSURE(context, x >= 0 && x < static_cast<int>(s.examples.size()));
    const int len = std::min(lengths->data.i32[i], width);
    auto first = pieces.begin() + static_cast<size_t>(i) * width;
    s.examples[x].hypothesis.assign(first, first + len);
    s.has_score = true;
    s.score += scores->data.f[i];
  }
  std::vector<std::string> out;
  for (const Sentence& s : sentences) out.push_back(Serialize(s));
  WriteStrings(out, Output(context, node, 0), Dims({static_cast<int>(out.size())}));
  return kTfLiteOk;
}

// ---- TokenPostProcess: resolves <unk> to the input character it stands for ----

TfLiteStatus PostProcessEval(TfLiteContext* context, TfLiteNode* node) {
  std::vector<std::string> out;
  for (const auto& in : Strings(Input(context, node, 0))) {
    Sentence s = Parse(in);
    for (Example& e : s.examples) {
      for (size_t i = 0; i < e.hypothesis.size(); ++i) {
        if (e.hypothesis[i] == "<unk>" && i < e.input.size()) e.hypothesis[i] = e.input[i];
      }
    }
    out.push_back(Serialize(s));
  }
  WriteStrings(out, Output(context, node, 0), Dims({static_cast<int>(out.size())}));
  return kTfLiteOk;
}

TfLiteRegistration* Register(TfLiteStatus (*eval)(TfLiteContext*, TfLiteNode*), bool token_options) {
  auto* r = new TfLiteRegistration{};
  if (token_options) {
    r->init = TokenOptionsInit;
    r->free = TokenOptionsFree;
  }
  r->prepare = DynamicOutputs;
  r->invoke = eval;
  return r;
}

}  // namespace

bool RoadrunnerCapitalizer::Init(const std::string& model_path, int window, int left, int right,
                                 std::string* error) {
  window_ = window;
  left_ = left;
  right_ = right;
  if (window_ <= left_ + right_) {
    *error = "bad capitalization window";
    return false;
  }
  model_ = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (!model_) {
    *error = "no capitalization model at " + model_path;
    return false;
  }
  static TfLiteRegistration* const input_context = Register(InputContextEval, true);
  static TfLiteRegistration* const retokenize = Register(RetokenizeEval, false);
  static TfLiteRegistration* const token_tensors = Register(TokenTensorsEval, true);
  static TfLiteRegistration* const attach = Register(AttachHypothesisEval, false);
  static TfLiteRegistration* const post_process = Register(PostProcessEval, false);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  AddPqrnnOps(&resolver);
  resolver.AddCustom("SentenceToInputContext", input_context);
  resolver.AddCustom("RetokenizeSentence", retokenize);
  resolver.AddCustom("SentenceToTokenTensors", token_tensors);
  resolver.AddCustom("AttachHypothesisToSentence", attach);
  resolver.AddCustom("TokenPostProcess", post_process);
  if (tflite::InterpreterBuilder(*model_, resolver)(&interpreter_) != kTfLiteOk || !interpreter_ ||
      interpreter_->inputs().size() != 1 || interpreter_->outputs().empty()) {
    *error = "cannot build capitalization interpreter";
    return false;
  }
  return true;
}

std::string RoadrunnerCapitalizer::Run(const std::string& sentence) {
  TfLiteTensor* in = interpreter_->tensor(interpreter_->inputs()[0]);
  tflite::DynamicBuffer buf;
  buf.AddString(sentence.data(), sentence.size());
  buf.WriteToTensor(in, Dims({1}));
  if (interpreter_->AllocateTensors() != kTfLiteOk || interpreter_->Invoke() != kTfLiteOk) return "";
  const TfLiteTensor* out = interpreter_->tensor(interpreter_->outputs()[0]);
  if (out->type != kTfLiteString || tflite::GetStringCount(out) < 1) return "";
  const tflite::StringRef s = tflite::GetString(out, 0);
  return std::string(s.str, s.len);
}

void RoadrunnerCapitalizer::Apply(std::vector<std::string>* words) {
  const int n = static_cast<int>(words->size());
  std::vector<std::string> cased = *words;
  const int step = window_ - left_ - right_;
  for (int start = 0; start < n; start += step) {
    const int end = std::min(n, start + window_);
    const bool last = end == n;
    // The words this window decides: all but its buffers, except at the ends.
    const int keep_from = start == 0 ? 0 : start + left_;
    const int keep_to = last ? n : end - right_;
    Sentence s;
    s.has_text = true;
    ProtoWriter request;
    for (int i = start; i < end; ++i) {
      if (i > start) s.original_text += ' ';
      const int left = static_cast<int>(s.original_text.size());
      s.original_text += (*words)[i];
      s.context.push_back((*words)[i]);
      s.spans.emplace_back(left, static_cast<int>(s.original_text.size()));
      request.Bytes(1, (*words)[i]);
    }
    request.Bytes(2, s.original_text);
    ProtoWriter input_request;
    input_request.Message(3, request);
    s.input_request = input_request.data();
    const Sentence out = Parse(Run(Serialize(s)));
    for (const Example& e : out.examples) {
      const int i = start + e.span_left;
      if (e.span_right != e.span_left + 1 || i < keep_from || i >= keep_to) continue;
      if (e.hypothesis.empty() || (e.hypothesis.size() == 1 && e.hypothesis[0] == "<self>")) continue;
      std::string word;
      for (const auto& p : e.hypothesis) word += p;
      cased[i] = word;
    }
    if (last) break;
  }
  *words = std::move(cased);
}

}  // namespace soda

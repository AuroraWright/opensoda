#include "punctuator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <utility>

#include "config.h"
#include "flatbuffers/flexbuffers.h"
#include "proto_wire.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "unicode_case.h"

namespace soda {
namespace {

// ---- seq_flow_lite SEQUENCE_STRING_PROJECTION (murmur hash, ternary bits) ----

constexpr uint64_t kMul = 0xc6a4a7935bd1e995ULL;
constexpr uint64_t kMul2 = 0x9e3779b97f4a7835ULL;

uint64_t ShiftMix(uint64_t v) { return v ^ (v >> 47); }

// Matches the reference, which ORs in plain (signed) chars.
uint64_t Load64VariableLength(const char* p, int len) {
  uint64_t v = 0;
  --len;
  do {
    v = (v << 8) | static_cast<uint64_t>(static_cast<int64_t>(p[len]));
  } while (--len >= 0);
  return v;
}

void MurmurHash128(const std::string& s, uint64_t* low, uint64_t* high) {
  const size_t len = s.size();
  uint64_t h1 = len * kMul;
  uint64_t h2 = 0;
  const size_t aligned = len & ~size_t{7};
  for (size_t i = 0; i < aligned; i += 8) {
    h1 ^= ShiftMix(Load64VariableLength(s.data() + i, 8) * kMul) * kMul;
    h1 *= kMul;
    h2 ^= h1;
  }
  if (len & 7) {
    h1 ^= Load64VariableLength(s.data() + aligned, len & 7);
    h1 *= kMul;
    h2 ^= h1;
  }
  h1 = ShiftMix(h1) * kMul;
  h2 ^= h1;
  h1 = ShiftMix(h1);
  h2 = ShiftMix(h2 * kMul2) * kMul2;
  *low = h1;
  *high = h2;
}

std::vector<uint64_t> HashCodes(const std::string& word, int feature_size) {
  std::vector<uint64_t> codes;
  uint64_t low = 0, high = 0;
  for (int i = 0; i < feature_size; i += 64) {
    if (i == 0) {
      MurmurHash128(word, &low, &high);
    } else {
      uint64_t h = ShiftMix(low) * kMul;
      uint64_t h2 = high ^ h;
      high = ShiftMix(h);
      low = ShiftMix(h2 * kMul2) * kMul2;
    }
    codes.push_back(low);
    codes.push_back(high);
  }
  return codes;
}

// Lowercases ASCII, Latin-1 and fullwidth Latin letters (the cases u_tolower
// changes in text this model sees).
std::string LowerCase(const std::string& in) {
  std::string out;
  for (size_t i = 0; i < in.size();) {
    const unsigned char c = in[i];
    if (c < 0x80) {
      out += static_cast<char>(c >= 'A' && c <= 'Z' ? c + 32 : c);
      ++i;
    } else if (c == 0xC3 && i + 1 < in.size() &&
               static_cast<unsigned char>(in[i + 1]) >= 0x80 &&
               static_cast<unsigned char>(in[i + 1]) <= 0x9E &&
               static_cast<unsigned char>(in[i + 1]) != 0x97) {
      out += static_cast<char>(c);
      out += static_cast<char>(in[i + 1] + 0x20);
      i += 2;
    } else if (c == 0xEF && i + 2 < in.size() &&
               static_cast<unsigned char>(in[i + 1]) == 0xBC &&
               static_cast<unsigned char>(in[i + 2]) >= 0xA1 &&
               static_cast<unsigned char>(in[i + 2]) <= 0xBA) {
      // U+FF21..FF3A -> U+FF41..FF5A
      const int lower = 0xFF41 + (static_cast<unsigned char>(in[i + 2]) - 0xA1);
      out += static_cast<char>(0xEF);
      out += static_cast<char>(0xBD);
      out += static_cast<char>(0x80 | (lower & 0x3F));
      i += 3;
    } else {
      const int n = c < 0xE0 ? 2 : c < 0xF0 ? 3 : 4;
      out.append(in, i, n);
      i += n;
    }
  }
  return out;
}

struct ProjectionParams {
  int feature_size = 0;
  bool add_bos = false;
  bool add_eos = true;
};

void* ProjectionInit(TfLiteContext*, const char* buffer, size_t length) {
  auto* p = new ProjectionParams;
  const auto m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length).AsMap();
  p->feature_size = m["feature_size"].AsInt32();
  p->add_bos = !m["add_bos_tag"].IsNull() && m["add_bos_tag"].AsBool();
  p->add_eos = m["add_eos_tag"].IsNull() || m["add_eos_tag"].AsBool();
  // The packs ship murmur-hashed, space-split, unrestricted-vocabulary models;
  // other options would change the features, so refuse them.
  const std::string hash = m["hashtype"].IsNull() ? "murmur" : m["hashtype"].AsString().str();
  if (hash != "murmur" || !m["split_on_space"].AsBool() ||
      m["vocabulary"].AsString().size() != 0 || m["word_novelty_bits"].AsInt32() != 0 ||
      m["doc_size_levels"].AsInt32() != 0 || m["add_first_cap_feature"].AsFloat() != 0 ||
      m["add_all_caps_feature"].AsFloat() != 0 || m["normalize_repetition"].AsBool() ||
      (!m["token_separators"].IsNull() && m["token_separators"].ToString().size() != 0)) {
    p->feature_size = -1;
  }
  return p;
}

void ProjectionFree(TfLiteContext*, void* buffer) {
  delete static_cast<ProjectionParams*>(buffer);
}

TfLiteStatus ProjectionPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* p = static_cast<ProjectionParams*>(node->user_data);
  if (p->feature_size <= 0) {
    context->ReportError(context, "unsupported SEQUENCE_STRING_PROJECTION options");
    return kTfLiteError;
  }
  TfLiteTensor* out = &context->tensors[node->outputs->data[0]];
  if (out->allocation_type != kTfLiteDynamic) {
    out->allocation_type = kTfLiteDynamic;
    out->data.raw = nullptr;
  }
  return kTfLiteOk;
}

TfLiteStatus ProjectionEval(TfLiteContext* context, TfLiteNode* node) {
  auto* p = static_cast<ProjectionParams*>(node->user_data);
  const TfLiteTensor* in = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* out = &context->tensors[node->outputs->data[0]];
  const tflite::StringRef text = tflite::GetString(in, 0);
  std::vector<std::string> words;
  for (size_t i = 0; i < static_cast<size_t>(text.len);) {
    while (i < static_cast<size_t>(text.len) && text.str[i] == ' ') ++i;
    size_t j = i;
    while (j < static_cast<size_t>(text.len) && text.str[j] != ' ') ++j;
    if (j > i) words.emplace_back(text.str + i, j - i);
    i = j;
  }
  if (p->add_bos) words.insert(words.begin(), "<BOS>");
  if (p->add_eos) words.push_back("<EOS>");
  if (words.empty()) return kTfLiteError;
  TfLiteIntArray* dims = TfLiteIntArrayCreate(3);
  dims->data[0] = 1;
  dims->data[1] = static_cast<int>(words.size());
  dims->data[2] = p->feature_size;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, out, dims));

  static const uint8_t kQuantized[4] = {127, 255, 0, 127};
  static const float kFloat[4] = {0.0f, 1.0f, -1.0f, 0.0f};
  for (size_t w = 0; w < words.size(); ++w) {
    const bool special = (p->add_bos && w == 0) || (p->add_eos && w + 1 == words.size());
    const std::vector<uint64_t> codes =
        HashCodes(special ? words[w] : LowerCase(words[w]), p->feature_size);
    const size_t base = w * p->feature_size;
    int k = 0;
    for (uint64_t hash : codes) {
      for (int kmax = std::min(k + 32, p->feature_size); k < kmax; ++k, hash >>= 2) {
        if (out->type == kTfLiteUInt8) {
          out->data.uint8[base + k] = kQuantized[hash & 3];
        } else {
          out->data.f[base + k] = kFloat[hash & 3];
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteRegistration* RegisterProjection() {
  static TfLiteRegistration r = {ProjectionInit, ProjectionFree, ProjectionPrepare,
                                 ProjectionEval};
  return &r;
}

// ---- seq_flow_lite QRNN pooling: state = state * f + z over time ----

float Dequantize(const TfLiteTensor* t, int i) {
  return (static_cast<int32_t>(t->data.uint8[i]) - t->params.zero_point) * t->params.scale;
}

uint8_t Quantize(float v, int32_t zero_point, float inverse_scale) {
  const float x = v * inverse_scale;
  const int32_t q = static_cast<int32_t>(x + (x >= 0 ? 0.5f : -0.5f)) + zero_point;
  return static_cast<uint8_t>(std::clamp(q, 0, 255));
}

TfLiteStatus PoolingPrepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* multiplier = &context->tensors[node->inputs->data[0]];
  TfLiteTensor* out = &context->tensors[node->outputs->data[0]];
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, out, TfLiteIntArrayCopy(multiplier->dims)));
  if (node->outputs->size == 2) {
    TfLiteIntArray* d = TfLiteIntArrayCreate(2);
    d->data[0] = 1;
    d->data[1] = multiplier->dims->data[2];
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, &context->tensors[node->outputs->data[1]], d));
  }
  return kTfLiteOk;
}

TfLiteStatus PoolingEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* multiplier = &context->tensors[node->inputs->data[0]];
  const TfLiteTensor* constant = &context->tensors[node->inputs->data[1]];
  const TfLiteTensor* direction = &context->tensors[node->inputs->data[2]];
  TfLiteTensor* out = &context->tensors[node->outputs->data[0]];
  const bool forward = direction->data.uint8[0] == 255;
  const int steps = multiplier->dims->data[1];
  const int size = multiplier->dims->data[2];
  std::vector<float> state(size, 0.0f);
  const float inv_scale = 1.0f / out->params.scale;
  for (int i = 0; i < steps; ++i) {
    const int t = forward ? i : steps - 1 - i;
    for (int j = 0; j < size; ++j) {
      const int idx = t * size + j;
      state[j] = state[j] * Dequantize(multiplier, idx) + Dequantize(constant, idx);
      out->data.uint8[idx] = Quantize(state[j], out->params.zero_point, inv_scale);
    }
  }
  if (node->outputs->size == 2) {
    TfLiteTensor* fs = &context->tensors[node->outputs->data[1]];
    for (int j = 0; j < size; ++j) {
      fs->data.uint8[j] = Quantize(state[j], fs->params.zero_point, 1.0f / fs->params.scale);
    }
  }
  return kTfLiteOk;
}

TfLiteRegistration* RegisterPooling() {
  static TfLiteRegistration r = {nullptr, nullptr, PoolingPrepare, PoolingEval};
  return &r;
}

// ---- speech/textnorm CharNGramFeatures: hashed character n-grams ----

// MurmurHash64A.
uint64_t Murmur64A(const char* p, size_t len, uint64_t seed) {
  constexpr uint64_t kMul = 0xC6A4A7935BD1E995ULL;
  uint64_t h = seed ^ (len * kMul);
  size_t i = 0;
  for (; i + 8 <= len; i += 8) {
    uint64_t k;
    std::memcpy(&k, p + i, 8);
    k *= kMul;
    k ^= k >> 47;
    k *= kMul;
    h ^= k;
    h *= kMul;
  }
  if (i < len) {
    uint64_t tail = 0;
    for (size_t j = 0; i + j < len; ++j) tail |= static_cast<uint64_t>(static_cast<uint8_t>(p[i + j])) << (8 * j);
    h ^= tail;
    h *= kMul;
  }
  h ^= h >> 47;
  h *= kMul;
  return h ^ (h >> 47);
}

// libsoda's default n-gram hash (its "Fingerprint2011"), for inputs of at
// most 32 bytes, which is all three-character n-grams need.
uint64_t Fingerprint2011Short(const char* p, size_t len) {
  constexpr uint64_t kMul = 0xC6A4A7935BD1E995ULL;
  uint64_t first = 0xA5B85C5E198ED849ULL, last = 0xA5B85C5E198ED849ULL;
  if (len >= 8) std::memcpy(&first, p, 8);
  if (len > 8) std::memcpy(&last, p + len - 8, 8);
  uint64_t x = Murmur64A(p, len, 0xEC9B9EE68CF8F8DDULL) + last;
  const uint64_t y = (x ^ first) * kMul;
  x = (x ^ (y >> 47) ^ y) * kMul;
  x = (x ^ (x >> 47)) * kMul;
  return x > 1 ? x : x | ~uint64_t{1};
}

struct CharNGramParams {
  int n = 0;
  int vocab_size = 0;
  bool output_text_index = false;
  std::string start_marker = "^";
  std::string end_marker = "$";
};

void* CharNGramInit(TfLiteContext*, const char* buffer, size_t length) {
  auto* p = new CharNGramParams;
  const auto m =
      flexbuffers::GetRoot(reinterpret_cast<const uint8_t*>(buffer), length).AsMap();
  // A serialized speech.textnorm.CharNGramFeaturesConfig: 1 start_marker,
  // 2 end_marker, 3 n, 4 vocab_size, 5 hash_function.
  const auto config = m["config"].AsString();
  ProtoReader r(std::string_view(config.c_str(), config.size()));
  while (r.Next()) {
    switch (r.field()) {
      case 1: p->start_marker = r.str(); break;
      case 2: p->end_marker = r.str(); break;
      case 3: p->n = static_cast<int>(r.int64()); break;
      case 4: p->vocab_size = static_cast<int>(r.int64()); break;
      case 5:
        // libsoda's default hash, also under this name; others are
        // unsupported.
        if (!r.bytes().empty() && r.bytes() != "Fingerprint2011ExternalUsage") p->n = -1;
        break;
    }
  }
  p->output_text_index = m["output_text_index"].AsBool();
  return p;
}

void CharNGramFree(TfLiteContext*, void* buffer) { delete static_cast<CharNGramParams*>(buffer); }

TfLiteStatus CharNGramPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* p = static_cast<CharNGramParams*>(node->user_data);
  if (p->n <= 0 || node->inputs->size != 1 || node->outputs->size != 3) {
    context->ReportError(context, "unsupported CharNGramFeatures options");
    return kTfLiteError;
  }
  for (int i = 0; i < 2; ++i) {
    TfLiteTensor* out = &context->tensors[node->outputs->data[i]];
    out->allocation_type = kTfLiteDynamic;
  }
  return context->ResizeTensor(context, &context->tensors[node->outputs->data[2]],
                               TfLiteIntArrayCreate(0));
}

// For each input string, the ids of its 1- to n-character n-grams (of the
// string between the markers), in order of start then length, and the most
// ids any string had. With output_text_index, each id comes with the index
// of the string it came from; otherwise with its place in a [strings, most]
// layout.
TfLiteStatus CharNGramEval(TfLiteContext* context, TfLiteNode* node) {
  auto* p = static_cast<CharNGramParams*>(node->user_data);
  const TfLiteTensor* in = &context->tensors[node->inputs->data[0]];
  std::vector<int32_t> ids, text_index, position;
  int32_t most = 0;
  for (int t = 0, count = tflite::GetStringCount(in); t < count; ++t) {
    const tflite::StringRef ref = tflite::GetString(in, t);
    const std::string s = p->start_marker + std::string(ref.str, ref.len) + p->end_marker;
    std::vector<size_t> starts{0};  // Code point boundaries.
    for (size_t i = 0; i < s.size();) {
      const unsigned char c = s[i];
      i += std::min(s.size() - i, size_t{c < 0xC2 || c > 0xF4 ? 1u : c < 0xE0 ? 2u : c < 0xF0 ? 3u : 4u});
      starts.push_back(i);
    }
    const size_t before = ids.size();
    for (size_t i = 0; i + 1 < starts.size(); ++i) {
      for (size_t j = i + 1; j < starts.size() && j <= i + p->n; ++j) {
        uint32_t id = static_cast<uint32_t>(Fingerprint2011Short(s.data() + starts[i], starts[j] - starts[i])) & 0x7FFFFFFF;
        if (p->vocab_size > 0) id %= p->vocab_size;
        position.push_back(static_cast<int32_t>(ids.size() - before));
        ids.push_back(static_cast<int32_t>(id));
        text_index.push_back(t);
      }
    }
    most = std::max(most, static_cast<int32_t>(ids.size() - before));
  }
  if (!p->output_text_index) {
    for (size_t i = 0; i < text_index.size(); ++i) text_index[i] = text_index[i] * most + position[i];
  }
  for (int i = 0; i < 2; ++i) {
    TfLiteIntArray* dims = TfLiteIntArrayCreate(1);
    dims->data[0] = static_cast<int>(ids.size());
    TfLiteTensor* out = &context->tensors[node->outputs->data[i]];
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, out, dims));
    std::copy(i == 0 ? ids.begin() : text_index.begin(), i == 0 ? ids.end() : text_index.end(),
              out->data.i32);
  }
  context->tensors[node->outputs->data[2]].data.i32[0] = most;
  return kTfLiteOk;
}

TfLiteRegistration* RegisterCharNGram() {
  static TfLiteRegistration r = {CharNGramInit, CharNGramFree, CharNGramPrepare, CharNGramEval};
  return &r;
}

}  // namespace

void AddPqrnnOps(tflite::MutableOpResolver* resolver) {
  resolver->AddCustom("SEQUENCE_STRING_PROJECTION", RegisterProjection());
  resolver->AddCustom("pooling_op", RegisterPooling());
  resolver->AddCustom("PoolingOp", RegisterPooling());
  resolver->AddCustom("CharNGramFeatures", RegisterCharNGram());
}

bool Punctuator::Init(const std::string& models_dir, std::string* error) {
  const std::string path = models_dir + "/SODA_punctuation_model.tflite";
  model_ = tflite::FlatBufferModel::BuildFromFile(path.c_str());
  if (!model_) {
    *error = "no punctuation model at " + path;
    return false;
  }
  tflite::ops::builtin::BuiltinOpResolver resolver;
  AddPqrnnOps(&resolver);
  if (tflite::InterpreterBuilder(*model_, resolver)(&interpreter_) != kTfLiteOk ||
      !interpreter_ || interpreter_->AllocateTensors() != kTfLiteOk) {
    *error = "cannot build punctuation interpreter";
    return false;
  }
  for (int idx : interpreter_->outputs()) {
    // ja-JP names them "Softmax"; de-DE "pqrnn_model/pqrnn_encoder/Softmax".
    const char* name = interpreter_->tensor(idx)->name;
    if (const char* slash = std::strrchr(name, '/')) name = slash + 1;
    if (!std::strcmp(name, "Softmax")) scores_output_ = idx;
    if (!std::strcmp(name, "LabelMap")) labels_output_ = idx;
  }
  // A Roadrunner tagger returns int32 labels instead of class scores.
  const TfLiteTensor* out0 = interpreter_->tensor(interpreter_->outputs()[0]);
  roadrunner_ = scores_output_ < 0 && interpreter_->outputs().size() == 1 &&
                out0->type == kTfLiteInt32 &&
                interpreter_->tensor(interpreter_->inputs()[0])->type == kTfLiteString;
  if (scores_output_ < 0 && !roadrunner_) {
    *error = "unsupported punctuation model";
    return false;
  }
  // speech_formatting.UnspokenPunctuationInferenceSettings: field 7 is the
  // BasicTextSegmenterConfig, whose field 1 lists the supported marks.
  // Field 7 is the BasicTextSegmenterConfig (1 supported punctuation, 2 end
  // of sentence punctuation), 11 the RoadrunnerPunctuationNormalizerParams.
  // libsoda overrides field 4, the PositionalCapitalizationBehavior (see
  // Capitalize).
  std::string settings;
  if (ReadFile(models_dir + "/SODA_punctuation_config.pb", &settings)) {
    ProtoReader r(settings);
    while (r.Next()) {
      if (r.field() == 7) {
        ProtoReader seg(r.bytes());
        while (seg.Next()) {
          if (seg.field() == 1) supported_marks_.push_back(seg.str());
          if (seg.field() == 2) end_of_sentence_marks_.push_back(seg.str());
          if (seg.field() == 3) segment_by_ = static_cast<int>(seg.int64());
        }
      }
      if (r.field() != 11) continue;
      ProtoReader rr(r.bytes());
      while (rr.Next()) {
        switch (rr.field()) {
          case 2: codebook_.push_back(rr.str()); break;
          case 3: append_dummy_token_ = rr.int64() != 0; break;
          case 7: window_size_ = static_cast<int>(rr.int64()); break;
          case 8: window_left_ = static_cast<int>(rr.int64()); break;
          case 9: window_right_ = static_cast<int>(rr.int64()); break;
          case 10: window_min_remainder_ = static_cast<int>(rr.int64()); break;
        }
      }
    }
  }
  if (roadrunner_ && codebook_.empty()) {
    *error = "punctuation config lacks a codebook";
    return false;
  }
  return true;
}

// libsoda tags words in windows of window_size_ words that overlap by the
// left and right buffers, keeping each window's labels between its buffers;
// a last window shorter than window_min_remainder_ is merged into the one
// before.
std::vector<int> Punctuator::RoadrunnerLabels(const std::vector<std::string>& tokens) {
  struct Window { int start, end, keep_start, keep_end; };
  const int n = static_cast<int>(tokens.size());
  const int size = std::max(1, window_size_);
  const int left = std::min(window_left_, size - 1);
  const int right = std::min(window_right_, size - 1 - left);
  const int stride = size - left - right;
  std::vector<Window> windows;
  for (int s = 0; s < n; s += stride) {
    const int end = std::min(n, s + size);
    windows.push_back({s, end, s > 0 ? left : 0, end - s - (n > s + size ? right : 0)});
    if (n <= s + size) break;
  }
  if (windows.size() >= 2 && windows.back().end - windows.back().start < window_min_remainder_) {
    windows.pop_back();
    windows.back().end = n;
    windows.back().keep_end = n - windows.back().start;
  }
  std::vector<int> labels(n, 0);
  const int input = interpreter_->inputs()[0];
  for (const Window& w : windows) {
    tflite::DynamicBuffer buf;
    for (int i = w.start; i < w.end; ++i) buf.AddString(tokens[i].data(), tokens[i].size());
    if (interpreter_->ResizeInputTensor(input, {1, w.end - w.start}) != kTfLiteOk ||
        interpreter_->AllocateTensors() != kTfLiteOk) {
      break;
    }
    buf.WriteToTensor(interpreter_->tensor(input), nullptr);
    if (interpreter_->Invoke() != kTfLiteOk) break;
    const TfLiteTensor* out = interpreter_->tensor(interpreter_->outputs()[0]);
    const int count = static_cast<int>(out->bytes / sizeof(int32_t));
    for (int i = w.keep_start; i < w.keep_end && i < count; ++i) labels[w.start + i] = out->data.i32[i];
  }
  return labels;
}

void Punctuator::PunctuateRoadrunner(std::vector<std::string>* words, bool final) {
  std::vector<std::string> tokens;
  for (const auto& w : *words) tokens.push_back(LowerCase(w));
  if (append_dummy_token_) tokens.emplace_back();
  const std::vector<int> labels = RoadrunnerLabels(tokens);
  // A label is the mark before its word: (label - 1) / 2 indexes the
  // codebook, whose entries ("|.", "|,", ...) are the mark after a "|".
  for (size_t i = 0; i + 1 < labels.size() && i < words->size(); ++i) {
    if (!final && i + 1 == words->size()) break;  // Partials end unmarked.
    const int k = (labels[i + 1] - 1) >> 1;
    if (k < 0 || k >= static_cast<int>(codebook_.size())) continue;
    const std::string& entry = codebook_[k];
    const std::string mark = entry.substr(entry.rfind('|') + 1);
    std::string& word = (*words)[i];
    // libsoda leaves a word that already ends with a mark of the codebook
    // (checked in codebook order, up to its first empty entry) as it is.
    bool marked = false;
    for (const std::string& e : codebook_) {
      const std::string m = e.substr(e.rfind('|') + 1);
      if (m.empty()) break;
      if (word.size() >= m.size() && word.compare(word.size() - m.size(), m.size(), m) == 0) {
        marked = true;
        break;
      }
    }
    if (!marked) word += mark;
  }
}

// Upper-cases a word's first letter: ASCII, Latin-1, Latin Extended-A and
// Cyrillic, enough for the packs' languages.
static void UpcaseFirst(std::string* w) {
  if (w->empty()) return;
  const unsigned char c0 = (*w)[0];
  if (c0 >= 'a' && c0 <= 'z') {
    (*w)[0] = static_cast<char>(c0 - 32);
    return;
  }
  if (c0 < 0xC0 || c0 >= 0xE0 || w->size() < 2) return;
  uint32_t c = ((c0 & 0x1F) << 6) | (static_cast<unsigned char>((*w)[1]) & 0x3F);
  if ((c >= 0xE0 && c <= 0xFE && c != 0xF7) || (c >= 0x430 && c <= 0x44F)) {
    c -= 0x20;
  } else if (c >= 0x450 && c <= 0x45F) {
    c -= 0x50;
  } else if (c >= 0x100 && c <= 0x17F && c != 0x131 && c != 0x138 && c != 0x149 && c != 0x17F &&
             ((c < 0x139 || c > 0x148) && (c < 0x179) ? c % 2 == 1 : c % 2 == 0)) {
    c -= 1;
  } else {
    return;
  }
  (*w)[0] = static_cast<char>(0xC0 | (c >> 6));
  (*w)[1] = static_cast<char>(0x80 | (c & 0x3F));
}

// libsoda's alphanumeric test for where capitalization applies: digits,
// ASCII letters and the letter blocks of 2- and 3-byte UTF-8 (4-byte
// characters never match).
static bool IsAlnumAt(const std::string& w, size_t i) {
  const unsigned char c0 = w[i];
  uint32_t c = c0;
  if (c0 >= 0x80) {
    if ((c0 & 0xE0) == 0xC0 && i + 1 < w.size()) {
      c = ((c0 & 0x1F) << 6) | (static_cast<unsigned char>(w[i + 1]) & 0x3F);
    } else if ((c0 & 0xF0) == 0xE0 && i + 2 < w.size()) {
      c = ((c0 & 0x0F) << 12) | ((static_cast<unsigned char>(w[i + 1]) & 0x3F) << 6) |
          (static_cast<unsigned char>(w[i + 2]) & 0x3F);
    } else {
      return false;
    }
  }
  constexpr std::pair<uint32_t, uint32_t> kRanges[] = {
      {'0', 10},   {'A', 26},   {'a', 26},  {0xC0, 23},   {0xD8, 31},   {0xF8, 449},
      {0x389, 249}, {0x48A, 255}, {0x5C8, 39}, {0x620, 43}, {0x66E, 104}, {0x710, 32},
      {0x74B, 212}, {0x1CFB, 182}, {0x1DFB, 246}, {0xFB00, 7}, {0x370, 4}, {0xB5, 1},
      {0x376, 1}};
  for (const auto& [start, n] : kRanges) {
    if (c - start < n) return true;
  }
  return false;
}

void Punctuator::Capitalize(std::vector<std::string>* words) const {
  // libsoda sets every pack's PositionalCapitalizationBehavior to
  // CAPITALIZE_ALL, whatever SODA_punctuation_config.pb says (ja-JP has
  // none): the first model token is capitalized, and each one after a token
  // that ends a sentence.
  auto is = [](const std::vector<std::string>& marks, std::string_view t) {
    return std::find(marks.begin(), marks.end(), t) != marks.end();
  };
  bool next = true;
  if (segment_by_ != kSimpleWhitespace) {
    // A token is a character, with the marks that follow it, so a mark inside
    // a word capitalizes the next character (ja-JP: `シャンT。Faceb` from
    // `シャンTfaceb`).
    for (auto& w : *words) {
      for (size_t i = 0; i < w.size();) {
        std::string ch = w.substr(i, CharLength(w, i));
        if (ch == " ") {
        } else if (is(supported_marks_, ch)) {
          next = is(end_of_sentence_marks_, ch);
        } else {
          if (next) {
            UpcaseFirst(&ch);
            w.replace(i, ch.size(), ch);
          }
          next = false;
        }
        i += ch.size();
      }
    }
    return;
  }
  for (auto& w : *words) {
    if (next) {
      // At the first letter or digit ("#urlaub" → "#Urlaub", "(1)" unchanged).
      size_t b = 0;
      while (b < w.size() && !IsAlnumAt(w, b)) b += CharLength(w, b);
      std::string rest = w.substr(b);
      UpcaseFirst(&rest);
      w = w.substr(0, b) + rest;
    }
    next = false;
    for (const auto& m : end_of_sentence_marks_) {
      next |= w.size() >= m.size() && w.compare(w.size() - m.size(), m.size(), m) == 0;
    }
  }
}

void Punctuator::Punctuate(std::vector<std::string>* words, bool final) {
  if (roadrunner_) {
    PunctuateRoadrunner(words, final);
    Capitalize(words);
    return;
  }
  auto is_mark = [&](std::string_view t) {
    return std::find(supported_marks_.begin(), supported_marks_.end(), t) != supported_marks_.end();
  };
  // Model tokens, and for each the (word, byte offset) it ends at: words
  // (SIMPLE_WHITESPACE, de-DE) or characters (UNICODE_CHARACTER, ja-JP).
  std::vector<std::string> tokens;
  std::vector<std::pair<size_t, size_t>> ends;
  for (size_t w = 0; w < words->size(); ++w) {
    const std::string& word = (*words)[w];
    if (segment_by_ == kSimpleWhitespace) {
      // The model sees each part of the word between spaces ("soir ?" from
      // French spoken punctuation) without the marks it already carries, and
      // a part of marks only not at all.
      for (size_t i = 0, j; i < word.size(); i = j + 1) {
        j = std::min(word.find(' ', i), word.size());
        size_t b = i, e = j;
        for (bool more = true; more;) {
          more = false;
          for (const auto& m : supported_marks_) {
            if (e - b >= m.size() && word.compare(b, m.size(), m) == 0) b += m.size(), more = true;
            if (e - b >= m.size() && word.compare(e - m.size(), m.size(), m) == 0) e -= m.size(), more = true;
          }
        }
        if (b == e) continue;
        tokens.push_back(word.substr(b, e - b));
        ends.emplace_back(w, j);
      }
      continue;
    }
    for (size_t i = 0; i < word.size();) {
      const unsigned char c = word[i];
      const size_t n = std::min(word.size() - i, size_t{c < 0x80 ? 1u : c < 0xE0 ? 2u : c < 0xF0 ? 3u : 4u});
      std::string ch = word.substr(i, n);
      i += n;
      if (is_mark(ch)) continue;  // Existing punctuation is not a model token.
      tokens.push_back(std::move(ch));
      ends.emplace_back(w, i);
    }
  }
  std::vector<std::string> marks = Predict(tokens);
  if (!final && !marks.empty()) marks.back().clear();
  // Insert from the back so earlier offsets stay valid.
  for (size_t k = marks.size(); k-- > 0;) {
    if (marks[k].empty()) continue;
    std::string& word = (*words)[ends[k].first];
    const size_t at = ends[k].second;
    bool marked = false;  // Already followed by a mark (de-DE: spoken; fr-FR: after a space)?
    const size_t next = std::min(word.find_first_not_of(' ', at), word.size());
    for (const auto& m : supported_marks_) {
      marked |= word.compare(next, m.size(), m) == 0 ||
                (segment_by_ == kSimpleWhitespace && at >= m.size() &&
                 word.compare(at - m.size(), m.size(), m) == 0);
    }
    if (!marked) word.insert(at, marks[k]);
  }
  Capitalize(words);
}

std::vector<std::string> Punctuator::Predict(const std::vector<std::string>& tokens,
                                             std::vector<std::vector<float>>* scores) {
  std::vector<std::string> marks(tokens.size());
  if (tokens.empty()) return marks;
  std::string text;
  for (const auto& w : tokens) {
    if (!text.empty()) text += ' ';
    text += w;
  }
  tflite::DynamicBuffer buf;
  buf.AddString(text.data(), text.size());
  buf.WriteToTensorAsVector(interpreter_->tensor(interpreter_->inputs()[0]));
  if (interpreter_->Invoke() != kTfLiteOk) return marks;

  if (labels_.empty() && labels_output_ >= 0) {
    const TfLiteTensor* lt = interpreter_->tensor(labels_output_);
    for (int i = 0, n = tflite::GetStringCount(lt); i < n; ++i) {
      const tflite::StringRef s = tflite::GetString(lt, i);
      // Labels are "KEEP" (no mark) or "KEEP|<mark>".
      const std::string label(s.str, s.len);
      const size_t bar = label.find('|');
      labels_.push_back(bar == std::string::npos ? "" : label.substr(bar + 1));
    }
  }
  const TfLiteTensor* st = interpreter_->tensor(scores_output_);
  const int classes = st->dims->data[st->dims->size - 1];
  int rows = 1;
  for (int d = 0; d + 1 < st->dims->size; ++d) rows *= st->dims->data[d];
  auto score = [&](int i) {
    return st->type == kTfLiteUInt8 ? (st->data.uint8[i] - st->params.zero_point) * st->params.scale
                                    : st->data.f[i];
  };
  if (scores) {
    scores->assign(tokens.size(), std::vector<float>(classes));
    for (int r = 0; r < rows && r < static_cast<int>(tokens.size()); ++r) {
      for (int c = 0; c < classes; ++c) (*scores)[r][c] = score(r * classes + c);
    }
  }
  for (int r = 0; r < rows && r < static_cast<int>(tokens.size()); ++r) {
    int best = 0;
    for (int c = 1; c < classes; ++c) {
      if (score(r * classes + c) > score(r * classes + best)) best = c;
    }
    if (best < static_cast<int>(labels_.size())) marks[r] = labels_[best];
  }
  return marks;
}

}  // namespace soda

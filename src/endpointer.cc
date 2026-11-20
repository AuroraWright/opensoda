#include "endpointer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/register.h"

namespace soda {
namespace {

// The "Relu1" custom op of ko-KR's and hi-IN's VAD models (libsoda's
// speech/pie/tflite/kernels/relu1.cc): clamps to [0, 1], unlike TFLite's
// RELU_N1_TO_1.
TfLiteStatus Relu1Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr && output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  output->type = input->type;
  return context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus Relu1Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = tflite::GetInput(context, node, 0);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);
  const float* in = input->data.f;
  float* out = output->data.f;
  for (int i = 0, n = tflite::NumElements(input); i < n; ++i) {
    float v = in[i];
    if (v <= 0.0f) v = 0.0f;
    if (v > 1.0f) v = 1.0f;
    out[i] = v;
  }
  return kTfLiteOk;
}

TfLiteRegistration* RegisterRelu1() {
  static TfLiteRegistration r = {nullptr, nullptr, Relu1Prepare, Relu1Eval};
  return &r;
}

}  // namespace

bool VoiceActivityDetector::Init(const std::string& models_dir, const VadConfig& config,
                                 std::string* error) {
  config_ = config;
  const std::string path = models_dir + "/" + config.model_file;
  if (!ReadFile(path, &model_data_) || model_data_.size() < 8) {
    *error = "cannot read " + path;
    return false;
  }
  // A flatbuffer carries its file identifier "TFL3" at bytes 4..7; recover
  // the XOR key from it.
  const unsigned char key = static_cast<unsigned char>(model_data_[4]) ^ 'T';
  if (key != 0) {
    for (char& c : model_data_) c = static_cast<char>(static_cast<unsigned char>(c) ^ key);
  }
  if (std::memcmp(model_data_.data() + 4, "TFL3", 4) != 0) {
    *error = path + " is not a (possibly obfuscated) TFLite model";
    return false;
  }
  model_ = tflite::FlatBufferModel::BuildFromBuffer(model_data_.data(), model_data_.size());
  tflite::ops::builtin::BuiltinOpResolver resolver;
  resolver.AddCustom("Relu1", RegisterRelu1());
  if (!model_ || tflite::InterpreterBuilder(*model_, resolver)(&interpreter_) != kTfLiteOk ||
      !interpreter_ || interpreter_->AllocateTensors() != kTfLiteOk) {
    *error = "cannot build interpreter for " + path;
    return false;
  }
  const TfLiteTensor* in = interpreter_->input_tensor(0);
  input_dim_ = in->dims->data[in->dims->size - 1];

  std::vector<float> stddev;
  if (!LoadMeanStdDev(models_dir + "/" + config.mean_stddev_file, &mean_, &stddev, error)) {
    return false;
  }
  if (static_cast<int>(mean_.size()) != input_dim_) {
    *error = "VAD mean/stddev has " + std::to_string(mean_.size()) +
             " dims but the model expects " + std::to_string(input_dim_);
    return false;
  }
  stddev_.resize(stddev.size());
  for (size_t i = 0; i < stddev.size(); ++i) stddev_[i] = stddev[i] > 0 ? stddev[i] : 1.0f;
  return true;
}

void VoiceActivityDetector::Reset() {
  if (interpreter_) interpreter_->ResetVariableTensors();
  frames_since_reset_ = frames_below_ = 0;
}

float VoiceActivityDetector::Process(const std::vector<float>& frame) {
  float* in = interpreter_->typed_input_tensor<float>(0);
  const int n = std::min<int>(frame.size(), input_dim_);
  std::copy(frame.begin(), frame.begin() + n, in);
  std::fill(in + n, in + input_dim_, 0.0f);
  if (config_.num_clusters > 0 && config_.cluster_id >= 0 &&
      n + config_.cluster_id < input_dim_) {
    in[n + config_.cluster_id] = 1.0f;
  } else if (config_.num_clusters == 0 && config_.cluster_id >= 0 && n < input_dim_) {
    in[n] = static_cast<float>(config_.cluster_id);
  }
  // Divided, not multiplied by the reciprocal: libsoda's inputs are
  // bit-identical this way.
  for (int i = 0; i < input_dim_; ++i) in[i] = (in[i] - mean_[i]) / stddev_[i];
  if (interpreter_->Invoke() != kTfLiteOk) return 0.0f;
  const TfLiteTensor* out = interpreter_->output_tensor(0);
  const int classes = out->dims->data[out->dims->size - 1];
  const int dim = std::clamp(config_.endpointer.posterior_dimension, 0, classes - 1);
  const float posterior = std::exp(out->data.f[dim]);  // The model ends in LOG_SOFTMAX.
  if (config_.min_frames_before_reset > 0 &&
      ++frames_since_reset_ >= config_.min_frames_before_reset) {
    frames_below_ = posterior < config_.reset_threshold ? frames_below_ + 1 : 0;
    if (frames_below_ >= config_.reset_frames_below) Reset();
  }
  return posterior;
}

void Endpointer::Init(const EndpointerConfig& config) {
  config_ = config;
  // Mirrors greco's arithmetic, including its float rounding.
  const float period = config.frame_period;
  auto frames = [&](float seconds) { return static_cast<int>(std::rint(seconds / period)); };
  auto at_least = [&](float seconds) {
    return static_cast<int>(std::ceil(static_cast<float>(seconds / period + 0.01)));
  };
  onset_window_ = frames(config.onset_window);
  offset_window_ = frames(config.offset_window);
  speech_on_window_ = frames(config.speech_on_window);
  const int longest = std::max({onset_window_, offset_window_, speech_on_window_});
  ring_size_ = static_cast<size_t>(std::max(longest, 1));
  // Windows as long as the ring cover one frame less.
  if (onset_window_ == longest) --onset_window_;
  if (offset_window_ == longest) --offset_window_;
  if (speech_on_window_ == longest) --speech_on_window_;
  onset_detect_ = at_least(config.onset_detect_dur);
  onset_confirm_ = at_least(config.onset_confirm_dur);
  offset_threshold_ = std::max(
      static_cast<int>(std::floor((config.offset_window - config.offset_confirm_dur) / period + 0.01f)),
      0);
  on_maintain_ = std::max(static_cast<int>(std::floor(config.on_maintain_dur / period + 0.01f)), 1) - 1;
  offset_confirm_frames_ = frames(config.offset_confirm_dur);
  margin_ = frames(config.endpoint_margin);
  margin_before_ = frames(config.additional_margin_before_speech);
  margin_after_ = frames(config.additional_margin_after_speech);
  Reset();
}

void Endpointer::Reset() {
  history_.clear();
  frames_ = 0;
  state_ = kPreSpeech;
  frames_output_ = 0;
  last_output_ = false;
  onset_start_ = 0;
  pending_ = 0;
  end_frame_ = -1;
  decisions_.clear();
}

bool Endpointer::Emit(long last, bool speech) {
  if (last < frames_output_) return false;
  const bool ended = last_output_ && !speech;
  if (ended) end_frame_ = frames_output_;
  decisions_.insert(decisions_.end(), last + 1 - frames_output_, speech);
  frames_output_ = last + 1;
  last_output_ = speech;
  return ended;
}

long Endpointer::SpeechOutputBound(long frame) const {
  // Scanning back from the newest frame, find where each window's count of
  // "on" frames first exceeds its threshold.
  const int n = static_cast<int>(history_.size());
  auto scan = [&](int window, int threshold) {
    int on = 0;
    for (int k = 0; k < std::min(window, n); ++k) {
      if (history_[n - 1 - k] && ++on > threshold) return window - k;
    }
    return -1;
  };
  const int offset_pos = scan(offset_window_, offset_threshold_);
  const int on_pos = std::min(speech_on_window_, n) >= 1 ? scan(speech_on_window_, on_maintain_) + 1 : 0;
  const long extra = std::max(std::max(offset_pos, on_pos), 0);
  return std::max(extra + frame - offset_confirm_frames_ - 1, 0L) + margin_ + margin_after_;
}

int Endpointer::CountOn(int window) const {
  const size_t n = std::min<size_t>(std::max(window, 0), history_.size());
  return static_cast<int>(std::count(history_.end() - n, history_.end(), true));
}

bool Endpointer::Process(float speech_posterior) {
  const long frame = frames_;
  // greco does not record the decision of the very first frame.
  if (++frames_ >= 2) {
    history_.push_back(speech_posterior > config_.decision_threshold);
    if (history_.size() > ring_size_) history_.pop_front();
  }
  bool ended = false;
  // Speech owed after an offset is output as frames arrive.
  bool owing = pending_ > 0;
  if (owing && frames_output_ <= frame) {
    const long last = std::min(pending_ + frames_output_ - 1, frame);
    pending_ = std::max(frames_output_ + pending_ - last - 1, 0L);
    ended |= Emit(last, true);
    owing = pending_ > 0;
  }
  switch (state_) {
    case kPreSpeech: {
      const int on = CountOn(onset_window_);
      if (on >= onset_detect_) {
        const long before = std::max(frame - onset_detect_, -1L);
        state_ = kPossibleOnset;
        onset_start_ = before + 1;
        ended |= Emit(before - margin_ - margin_before_, false);
      } else {
        // Silence is output late, so that an onset can still claim it.
        ended |= Emit(std::max(frame - on, 1L) - 2 - margin_ - margin_before_, false);
      }
      break;
    }
    case kPossibleOnset: {
      const int on = CountOn(onset_window_);
      if (on >= onset_confirm_) {
        state_ = kSpeechPresent;
        ended |= Emit(std::max(onset_start_ - margin_ - margin_before_, 0L) - 1, false);
        ended |= Emit(std::min(frame, SpeechOutputBound(frame)), true);
      } else if (on <= onset_detect_) {
        state_ = kPreSpeech;
      }
      break;
    }
    case kSpeechPresent:
      if (CountOn(speech_on_window_) <= on_maintain_ && !config_.never_transition_to_silence) {
        state_ = kPossibleOffset;
      }
      if (!owing) ended |= Emit(std::min(frame, SpeechOutputBound(frame)), true);
      break;
    case kPossibleOffset: {
      const int on = CountOn(offset_window_);
      if (on <= offset_threshold_) {
        state_ = kPreSpeech;
        const long last = std::max(frame - offset_confirm_frames_ - 1, 0L) + margin_ + margin_after_;
        if (last >= frame) pending_ = last - frame;
        ended |= Emit(std::min(last, frame), true);
      } else if (on >= on_maintain_) {
        state_ = kSpeechPresent;
        ended |= Emit(std::min(frame, SpeechOutputBound(frame)), true);
      }
      break;
    }
  }
  return ended;
}

}  // namespace soda

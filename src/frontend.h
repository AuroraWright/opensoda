// Streaming acoustic frontend matching SODA's greco feature pipeline:
//
//   Framer -> Window (pre-emphasis, dither, Hann) -> Fft (|X|^2)
//   -> FilterBank (mel, sqrt-magnitude summation, log) -> FrameStacker
//   -> SubsampleStream -> FrameNormalizeStream -> AppendClusterIdStream
//   -> FrameStacker -> SubsampleStream -> encoder input frames.

#ifndef SODA_FRONTEND_H_
#define SODA_FRONTEND_H_

#include <deque>
#include <vector>

#include "config.h"

namespace soda {

// Port of the triangular mel filterbank used by greco's FilterBank stream
// (identical to TensorFlow's MfccMelFilterbank).
class MelFilterbank {
 public:
  void Init(int num_bins, double sample_rate, int channels, double lower_hz,
            double upper_hz);
  // `power` holds num_bins squared magnitudes; writes `channels` sums of
  // weighted magnitudes.
  void Compute(const std::vector<float>& power, std::vector<float>* out) const;

 private:
  int channels_ = 0;
  int start_index_ = 0;
  int end_index_ = 0;
  std::vector<float> center_mel_;
  std::vector<float> weights_;
  std::vector<int> band_mapper_;
};

// greco's FFT (speech/portable/fft): the conjugate-pair split-radix
// transform of djbfft. A radix-4 pass splits n points into a half and two
// quarters, which are transformed in turn, down to 16- and 8-point kernels.
// Following its order of operations gives bit-identical spectra.
// Returns |X[k]|^2 for k = 0..n/2 of a real frame.
class PowerSpectrum {
 public:
  void Init(int fft_size);
  void Compute(const std::vector<float>& frame, std::vector<float>* power);

 private:
  // In-place transform of n complex values (interleaved), with e^{+i} and
  // the outputs out of order.
  void Transform(float* a, int n) const;
  void Pass(float* a, int n) const;

  int n_ = 0;
  // Per pass size n: cos and sin of 2 pi k / n for k = 1 .. n/4 - 1.
  std::vector<std::vector<float>> twiddle_;
  std::vector<int> position_;  // Output index of each frequency 0..n/2.
  std::vector<float> buf_;
};

// Stacks context frames, then keeps every `subsample`-th stacked frame.
class FrameStacker {
 public:
  void Init(const FrameStackConfig& c, int dim);
  void Push(const std::vector<float>& frame, std::vector<std::vector<float>>* out);
  void Flush(std::vector<std::vector<float>>* out);
  void Reset();
  int output_dim() const { return dim_ * (cfg_.left_context + 1 + cfg_.right_context); }

 private:
  void Emit(std::vector<std::vector<float>>* out);

  FrameStackConfig cfg_;
  int dim_ = 0;
  std::deque<std::vector<float>> window_;  // Frames t-left .. t+right.
  bool started_ = false;
  long emitted_ = 0;
};

class Frontend {
 public:
  bool Init(const FrontendConfig& config, int sample_rate,
            const std::vector<float>& mean, const std::vector<float>& stddev,
            std::string* error);
  void Reset();

  // Consumes samples on the int16 scale; appends ready encoder frames.
  void Accept(const float* samples, size_t n, std::vector<std::vector<float>>* out);
  // Signals end of audio; emits frames held back for right context.
  void Flush(std::vector<std::vector<float>>* out);

  int output_dim() const { return stack2_.output_dim(); }
  // When set, receives a copy of every stacked filterbank frame before
  // normalization (the input of the VAD branch).
  void set_stacked_tap(std::vector<std::vector<float>>* tap) { stacked_tap_ = tap; }
  // When set, receives every frame's FFT energies, dithered with
  // noise_scale (negative: the main path's).
  void set_power_tap(std::vector<std::vector<float>>* tap, float noise_scale = -1.0f) {
    power_tap_ = tap;
    tap_noise_scale_ = noise_scale;
  }
  int num_fft_bins() const { return num_bins_; }

  // Gated mode places greco's SegmenterStream after the stacked filterbanks:
  // their frames wait until Release() decides each, oldest first, as speech
  // (passed on towards the encoder) or silence (dropped). EndSegment() ends a
  // segment, emitting the frames held back for right context.
  void set_gated(bool gated) { gated_ = gated; }
  size_t held() const { return held_.size(); }
  void Release(bool speech, std::vector<std::vector<float>>* out);
  void EndSegment(std::vector<std::vector<float>>* out);
  // When set, receives for each output frame the number of stacked
  // filterbank frames up to and including its newest one: its end time in
  // units of stacked frames.
  void set_output_end_sink(std::vector<long>* sink) { output_ends_ = sink; }
  // Milliseconds of audio covered by one stacked (gated) frame.
  float stacked_frame_ms() const {
    return cfg_.subsample_after_gate ? cfg_.frame_step_ms : cfg_.frame_step_ms * cfg_.stack1.subsample;
  }
  // Milliseconds of audio covered by one output frame.
  float output_frame_ms() const {
    return cfg_.frame_step_ms * cfg_.stack1.subsample * cfg_.stack2.subsample;
  }

 private:
  void ProcessFrame(const float* frame_with_pad,
                    std::vector<std::vector<float>>* out);
  // Pre-emphasizes, dithers and windows a frame into windowed_.
  void Window(const float* x, float noise_scale, unsigned noise_index);
  void PostStack1(std::vector<std::vector<float>>& stacked,
                  std::vector<std::vector<float>>* out);
  // Normalizes a stacked frame and passes it on through the second stacker.
  // end: the stacked frame count at the frame's end, its time.
  void Forward(std::vector<float> frame, long end, std::vector<std::vector<float>>* out);
  // With subsample_after_gate, keeps every stack1.subsample-th frame passed
  // on since the segment start.
  void Pass(std::vector<float> frame, std::vector<std::vector<float>>* out);
  void Stack2Flush(std::vector<std::vector<float>>* out);

  FrontendConfig cfg_;
  int frame_len_ = 0;
  int frame_step_ = 0;
  std::vector<float> window_;
  std::vector<float> noise_;
  PowerSpectrum fft_;
  MelFilterbank mel_;
  FrameStacker stack1_, stack2_;
  std::vector<float> mean_, stddev_;

  std::vector<float> pending_;  // Samples not yet framed.
  unsigned frame_index_ = 0;    // Frames since the stream start (dither phase).
  std::vector<float> windowed_, power_, mel_out_;
  std::vector<std::vector<float>>* stacked_tap_ = nullptr;
  std::vector<std::vector<float>>* power_tap_ = nullptr;
  float tap_noise_scale_ = -1.0f;
  std::vector<float> tap_power_;
  int num_bins_ = 0;
  long passed_ = 0;  // Frames passed on in this segment (subsample_after_gate).

  bool gated_ = false;
  std::deque<std::vector<float>> held_;  // Stacked frames awaiting Release().
  long stacked_frames_ = 0;              // Stacked frames produced.
  long forwarded_end_ = 0;               // End of the newest forwarded frame.
  std::vector<long>* output_ends_ = nullptr;
};

}  // namespace soda

#endif  // SODA_FRONTEND_H_

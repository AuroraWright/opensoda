#include "frontend.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace soda {
namespace {

// Size of greco's precomputed dither table, and the strides it is walked with.
constexpr int kNoiseTableSize = 16127;
constexpr unsigned kNoiseSampleStride = 613;
constexpr unsigned kNoiseFrameStride = 1301;

float HzToMel(double hz) { return static_cast<float>(1127.0 * std::log(hz / 700.0 + 1.0)); }

int SamplesFor(float ms, int sample_rate) {
  return static_cast<int>(std::lround(ms * 0.001 * sample_rate));
}

}  // namespace

void MelFilterbank::Init(int num_bins, double sample_rate, int channels,
                         double lower_hz, double upper_hz) {
  channels_ = channels;
  // Channel edges are equally spaced in mel between the band limits.
  center_mel_.resize(channels + 1);
  const float mel_low = HzToMel(lower_hz);
  const float mel_high = HzToMel(upper_hz);
  const float spacing = (mel_high - mel_low) / static_cast<float>(channels + 1);
  for (int i = 0; i < channels + 1; ++i) {
    center_mel_[i] = spacing * static_cast<float>(i + 1) + mel_low;
  }
  const float hz_per_bin = static_cast<float>(sample_rate * 0.5 / (num_bins - 1));
  start_index_ = static_cast<int>(static_cast<float>(lower_hz) / hz_per_bin + 1.5f);
  end_index_ = std::min(num_bins - 1,
                        static_cast<int>(static_cast<float>(upper_hz) / hz_per_bin));

  band_mapper_.assign(num_bins, -2);
  int channel = 0;
  for (int i = 0; i < num_bins; ++i) {
    if (i < start_index_ || i > end_index_) continue;
    const float mel = HzToMel(hz_per_bin * static_cast<float>(i));
    while (channel < channels && center_mel_[channel] < mel) ++channel;
    band_mapper_[i] = channel - 1;
  }

  weights_.assign(num_bins, 0.0f);
  for (int i = start_index_; i <= end_index_; ++i) {
    const float mel = HzToMel(hz_per_bin * static_cast<float>(i));
    const int c = band_mapper_[i];
    if (c >= 0) {
      weights_[i] = (center_mel_[c + 1] - mel) / (center_mel_[c + 1] - center_mel_[c]);
    } else {
      weights_[i] = (center_mel_[0] - mel) / (center_mel_[0] - mel_low);
    }
  }
}

void MelFilterbank::Compute(const std::vector<float>& power,
                            std::vector<float>* out) const {
  out->assign(channels_, 0.0f);
  for (int i = start_index_; i <= end_index_; ++i) {
    const float magnitude = std::sqrt(power[i]);
    const float weighted = magnitude * weights_[i];
    int c = band_mapper_[i];
    if (c >= 0) (*out)[c] += weighted;
    ++c;
    if (c < channels_) (*out)[c] += magnitude - weighted;
  }
}

namespace {

const float kSqrtHalf = static_cast<float>(std::cos(M_PI / 4));

// djbfft's 8-point kernel.
void Fft8(float* x) {
  const float h = kSqrtHalf;
  const float r8 = x[8], i8 = x[9], i0 = x[1];
  const float a0 = i0 - i8, a1 = i8 + i0;
  const float r12 = x[12], i12 = x[13], r4 = x[4], i4 = x[5];
  const float a2 = r4 - r12, a3 = r12 + r4;
  const float a4 = a0 - a2, a5 = a0 + a2;
  const float a6 = i4 - i12, a7 = i12 + i4;
  const float a8 = x[0] - r8, a9 = r8 + x[0];
  const float a10 = a8 - a6, a11 = a6 + a8;
  const float r10 = x[10], i10 = x[11], r2 = x[2], i2 = x[3];
  const float b0 = r2 - r10, b1 = r10 + r2;
  const float r14 = x[14], i14 = x[15], r6 = x[6], i6 = x[7];
  const float b2 = i6 - i14, b3 = i14 + i6;
  const float b4 = b0 - b2, b5 = b0 + b2;
  const float b6 = i2 - i10, b7 = i10 + i2;
  const float b8 = r6 - r14, b9 = r14 + r6;
  const float b10 = b6 - b8, b11 = b6 + b8;
  const float c0 = (b4 - b11) * h, c1 = (b4 + b11) * h;
  x[10] = a10 - c0;
  x[11] = a5 - c1;
  x[8] = a10 + c0;
  x[9] = a5 + c1;
  const float c2 = (b10 - b5) * h, c3 = (b5 + b10) * h;
  x[14] = a11 - c3;
  x[15] = a4 - c2;
  x[12] = a11 + c3;
  x[13] = a4 + c2;
  const float d0 = a9 - a3, d1 = a3 + a9, d2 = a1 - a7, d3 = a1 + a7;
  x[6] = d0 + (b7 - b3);
  x[7] = d2 - (b1 - b9);
  x[4] = d0 - (b7 - b3);
  x[5] = d2 + (b1 - b9);
  x[0] = d1 + (b1 + b9);
  x[1] = d3 + (b3 + b7);
  x[2] = d1 - (b1 + b9);
  x[3] = d3 - (b3 + b7);
}

// djbfft's 16-point kernel: a pass, two 4-point transforms and Fft8, with
// the compiler's interleaving of loads and stores kept (some values are
// stored and read back).
void Fft16(float* x) {
  const float c = static_cast<float>(std::cos(M_PI / 8));
  const float s = static_cast<float>(std::sin(M_PI / 8));
  const float h = kSqrtHalf;
  float v2 = x[16], v3 = x[17], v4 = x[1];
  float v5 = x[0] - v2, v6 = v2 + x[0];
  float v7 = x[8], v8 = x[9], v10 = x[25], v9 = x[26];
  float v11 = v5 - (v8 - v10), v12 = v5 + (v8 - v10), v13 = v10 + v8;
  float v14 = x[23], v15 = x[24];
  x[8] = v15 + v7;
  x[9] = v13;
  x[17] = (v7 - v15) + (v4 - v3);
  float v16 = (v4 - v3) - (v7 - v15);
  x[0] = v6;
  x[1] = v3 + v4;
  float v17 = x[2], v18 = x[3], v19 = x[17], v20 = x[18];
  float v21 = v17 - v20, v22 = v20 + v17;
  float v24 = x[10], v23 = x[11], v25 = x[27], v26 = x[28];
  float v27 = v21 - (v23 - v25), v28 = v21 + (v23 - v25);
  float v29 = v28 * c, v30 = v24 - v9;
  x[10] = v9 + v24;
  x[11] = v25 + v23;
  float v31 = v28 * s, v32 = x[19], v33 = x[20], v34 = v18 - v32;
  x[2] = v22;
  x[3] = v32 + v18;
  float v35 = v30 + (v18 - v32);
  x[18] = (v27 * c) - (s * v35);
  x[19] = (v27 * s) + (c * v35);
  float v36 = (c * (v34 - v30)) - v31;
  x[25] = v16;
  x[26] = v29 + (s * (v34 - v30));
  float v37 = x[4], v38 = x[5], v39 = v37 - v33, v40 = v33 + v37;
  float v42 = x[12], v41 = x[13], v43 = x[29], v44 = x[30];
  float v45 = v39 - (v41 - v43), v46 = v39 + (v41 - v43), v47 = v43 + v41, v48 = v42 - v26;
  x[12] = v26 + v42;
  x[13] = v47;
  float v49 = x[21], v50 = x[22];
  float v51 = v48 + (v38 - v49), v52 = (v38 - v49) - v48;
  x[4] = v40;
  x[5] = v49 + v38;
  float v53 = (v46 + v52) * h, v54 = (v52 - v46) * h;
  float v55 = (v45 - v51) * h, v56 = (v45 + v51) * h;
  float v57 = x[6], v58 = x[7], v59 = v57 - v50, v60 = v50 + v57;
  float v61 = x[31], v63 = x[14], v62 = x[15];
  float v64 = v59 - (v62 - v61), v65 = v59 + (v62 - v61);
  x[14] = v44 + v63;
  x[15] = v61 + v62;
  float v66 = v58 - v14;
  x[6] = v60;
  x[7] = v14 + v58;
  float v67 = (v63 - v44) + (v58 - v14), v68 = v66 - (v63 - v44);
  float v69 = (s * v64) - (c * v67), v70 = (c * v64) + (s * v67);
  float v71 = (s * v68) - (c * v65), v72 = (s * v65) + (c * v68);
  float v73 = v11 - v55, v74 = v11 + v55;
  float v75 = x[18], v76 = x[19];
  float v77 = v75 - v69, v78 = v69 + v75, v79 = v74 + v78, v80 = v74 - v78;
  float v81 = v19 - v56, v82 = v56 + v19;
  x[22] = v73 + (v76 - v70);
  x[23] = v81 - v77;
  x[20] = v73 - (v76 - v70);
  x[21] = v77 + v81;
  float v83 = v70 + v76;
  x[16] = v79;
  x[17] = v82 + v83;
  x[18] = v80;
  x[19] = v82 - v83;
  float v84 = v12 - v53, v85 = v12 + v53;
  float v87 = x[25], v86 = x[26];
  float v88 = v86 - v72, v89 = v72 + v86, v90 = v85 + v89;
  x[26] = v85 - v89;
  x[27] = v36;
  x[31] = (v87 - v54) - v88;
  float v91 = x[27];
  x[29] = v88 + (v87 - v54);
  x[30] = v84 + (v91 - v71);
  float v92 = v84 - (v91 - v71), v93 = v71 + v91;
  x[24] = v90;
  x[25] = (v54 + v87) + v93;
  x[27] = (v54 + v87) - v93;
  x[28] = v92;
  Fft8(x);
}

}  // namespace

void PowerSpectrum::Init(int fft_size) {
  n_ = fft_size;
  twiddle_.assign(n_ + 1, {});
  for (int n = 32; n <= n_; n *= 2) {
    for (int k = 1; k < n / 4; ++k) {
      twiddle_[n].push_back(static_cast<float>(std::cos(2 * M_PI * k / n)));
      twiddle_[n].push_back(static_cast<float>(std::sin(2 * M_PI * k / n)));
    }
  }
  // Find where each frequency lands from the transform of a unit impulse at
  // sample 1, whose output at frequency f is e^{2 pi i f / n}.
  buf_.assign(2 * n_, 0.0f);
  buf_[2] = 1.0f;
  Transform(buf_.data(), n_);
  position_.assign(n_ / 2 + 1, 0);
  for (int p = 0; p < n_; ++p) {
    const double turns = std::atan2(buf_[2 * p + 1], buf_[2 * p]) / (2 * M_PI);
    const int f = static_cast<int>(std::lround(turns * n_) % n_ + n_) % n_;
    // The transform uses e^{+i}; frequency k of e^{-i} is its -k.
    const int k = (n_ - f) % n_;
    if (k <= n_ / 2) position_[k] = p;
  }
}

void PowerSpectrum::Pass(float* a, int n) const {
  // Quarters A, B, C, D: A <- A + C, B <- B + D, and C and D get
  // (A - C) +- i (B - D) times w^k and w^-k. Above 512 points the second
  // half of the twiddles comes from the first by symmetry, and k = n/8 uses
  // sqrt(1/2) directly.
  const int q = n / 4, m = n / 8;
  const bool symmetric = n >= 1024;
  const std::vector<float>& w = twiddle_[n];
  for (int k = 0; k < q; ++k) {
    float* A = a + 2 * k;
    float* B = A + 2 * q;
    float* C = B + 2 * q;
    float* D = C + 2 * q;
    const float ar = A[0], ai = A[1], br = B[0], bi = B[1];
    const float cr = C[0], ci = C[1], dr = D[0], di = D[1];
    A[0] = ar + cr;
    A[1] = ai + ci;
    B[0] = br + dr;
    B[1] = bi + di;
    const float t0 = ar - cr, t1 = ai - ci, u0 = br - dr, u1 = bi - di;
    const float z1r = t0 - u1, z1i = u0 + t1, z3r = t0 + u1, z3i = t1 - u0;
    if (k == 0) {
      C[0] = z1r;
      C[1] = z1i;
      D[0] = z3r;
      D[1] = z3i;
    } else if (symmetric && k == m) {
      C[0] = (z1r - z1i) * kSqrtHalf;
      C[1] = (z1r + z1i) * kSqrtHalf;
      D[0] = (z3r + z3i) * kSqrtHalf;
      D[1] = (z3i - z3r) * kSqrtHalf;
    } else {
      float w0, w1;
      if (symmetric && k > m) {
        w0 = w[2 * (q - k - 1) + 1];
        w1 = w[2 * (q - k - 1)];
      } else {
        w0 = w[2 * (k - 1)];
        w1 = w[2 * (k - 1) + 1];
      }
      C[0] = z1r * w0 - z1i * w1;
      C[1] = z1r * w1 + w0 * z1i;
      D[0] = z3r * w0 + z3i * w1;
      D[1] = w0 * z3i - z3r * w1;
    }
  }
}

void PowerSpectrum::Transform(float* a, int n) const {
  if (n == 8) return Fft8(a);
  if (n == 16) return Fft16(a);
  Pass(a, n);
  Transform(a + n, n / 4);
  Transform(a + 3 * n / 2, n / 4);
  Transform(a, n / 2);
}

void PowerSpectrum::Compute(const std::vector<float>& frame,
                            std::vector<float>* power) {
  for (int i = 0; i < n_; ++i) {
    buf_[2 * i] = i < static_cast<int>(frame.size()) ? frame[i] : 0.0f;
    buf_[2 * i + 1] = 0.0f;
  }
  Transform(buf_.data(), n_);
  power->resize(n_ / 2 + 1);
  for (int k = 0; k <= n_ / 2; ++k) {
    const float re = buf_[2 * position_[k]], im = buf_[2 * position_[k] + 1];
    (*power)[k] = re * re + im * im;
  }
}

void FrameStacker::Init(const FrameStackConfig& c, int dim) {
  cfg_ = c;
  dim_ = dim;
  Reset();
}

void FrameStacker::Reset() {
  window_.clear();
  started_ = false;
  emitted_ = 0;
}

void FrameStacker::Emit(std::vector<std::vector<float>>* out) {
  if (emitted_++ % cfg_.subsample != 0) return;
  std::vector<float> stacked;
  stacked.reserve(output_dim());
  for (const auto& f : window_) stacked.insert(stacked.end(), f.begin(), f.end());
  out->push_back(std::move(stacked));
}

void FrameStacker::Push(const std::vector<float>& frame,
                        std::vector<std::vector<float>>* out) {
  if (!started_) {
    started_ = true;
    // Missing left context is zeros with frame_zero_padding, and copies of
    // the first frame otherwise, so output starts with the first input frame.
    for (int i = 0; i < cfg_.left_context; ++i) {
      if (cfg_.zero_padding) {
        window_.emplace_back(dim_, 0.0f);
      } else {
        window_.push_back(frame);
      }
    }
  }
  window_.push_back(frame);
  const size_t span = cfg_.left_context + 1 + cfg_.right_context;
  if (window_.size() == span) {
    Emit(out);
    window_.pop_front();
  }
}

void FrameStacker::Flush(std::vector<std::vector<float>>* out) {
  if (cfg_.zero_padding && started_) {
    // Each held-back frame gets zero frames as its missing right context.
    const size_t span = cfg_.left_context + 1 + cfg_.right_context;
    const int pending = static_cast<int>(window_.size()) - cfg_.left_context;
    for (int i = 0; i < pending; ++i) {
      window_.emplace_back(dim_, 0.0f);
      if (window_.size() == span) {
        Emit(out);
        window_.pop_front();
      }
    }
  }
  Reset();
}

bool Frontend::Init(const FrontendConfig& config, int sample_rate,
                    const std::vector<float>& mean,
                    const std::vector<float>& stddev, std::string* error) {
  cfg_ = config;
  frame_len_ = SamplesFor(cfg_.frame_size_ms, sample_rate);
  frame_step_ = SamplesFor(cfg_.frame_step_ms, sample_rate);
  const int window_len = SamplesFor(cfg_.window_size_ms, sample_rate);
  if (window_len != frame_len_ || frame_len_ <= 0 || frame_step_ <= 0) {
    *error = "unsupported framing configuration";
    return false;
  }
  if (cfg_.window_type != 1) {
    *error = "only HANNING windows are implemented";
    return false;
  }
  // HANNING: a Hann window sampled at half-sample offsets, with the step
  // rounded to float as greco does.
  window_.resize(window_len);
  const double step = static_cast<float>(2.0 * M_PI / static_cast<float>(window_len));
  for (int n = 0; n < window_len; ++n) {
    window_[n] = static_cast<float>(0.5 - std::cos((n + 0.5) * step) * 0.5);
  }
  // The dither table is drawn once from a default-seeded mt19937.
  noise_.resize(kNoiseTableSize);
  std::mt19937 gen;
  std::normal_distribution<double> normal(0.0, 1.0);
  for (float& v : noise_) v = static_cast<float>(normal(gen));

  int fft_size = std::max(SamplesFor(cfg_.fft_input_length_ms, sample_rate),
                          cfg_.min_fft_size);
  if (cfg_.fft_overdrive) fft_size <<= 1;
  int pow2 = 1;
  while (pow2 < fft_size) pow2 <<= 1;
  fft_.Init(pow2);
  num_bins_ = pow2 / 2 + 1;
  mel_.Init(num_bins_, sample_rate, cfg_.channel_count,
            cfg_.lower_band_limit, cfg_.upper_band_limit);

  FrameStackConfig stack1 = cfg_.stack1;
  if (cfg_.subsample_after_gate) stack1.subsample = 1;
  stack1_.Init(stack1, cfg_.channel_count);
  const int normalized_dim = stack1_.output_dim();
  // Some packs (en-US) share one table with the VAD, whose input also has a
  // one-hot cluster id; the stacked features use its first entries.
  if (static_cast<int>(mean.size()) < normalized_dim || mean.size() != stddev.size()) {
    *error = "mean/stddev size does not match the stacked feature size";
    return false;
  }
  mean_.assign(mean.begin(), mean.begin() + normalized_dim);
  stddev_.assign(stddev.begin(), stddev.begin() + normalized_dim);
  stack2_.Init(cfg_.stack2, normalized_dim + (cfg_.cluster_id >= 0 ? 1 : 0));
  Reset();
  return true;
}

void Frontend::Reset() {
  pending_.clear();
  frame_index_ = 0;
  held_.clear();
  stacked_frames_ = forwarded_end_ = 0;
  passed_ = 0;
  stack1_.Reset();
  stack2_.Reset();
}

void Frontend::Accept(const float* samples, size_t n,
                      std::vector<std::vector<float>>* out) {
  pending_.insert(pending_.end(), samples, samples + n);
  size_t start = 0;
  while (pending_.size() - start >= static_cast<size_t>(frame_len_) + 1) {
    ProcessFrame(&pending_[start], out);
    start += frame_step_;
  }
  pending_.erase(pending_.begin(), pending_.begin() + start);
}

void Frontend::Flush(std::vector<std::vector<float>>* out) {
  std::vector<std::vector<float>> stacked;
  stack1_.Flush(&stacked);
  PostStack1(stacked, out);
  // In gated mode the held frames still await decisions.
  if (!gated_) Stack2Flush(out);
  pending_.clear();
}

void Frontend::Release(bool speech, std::vector<std::vector<float>>* out) {
  if (held_.empty()) return;
  std::vector<float> frame = std::move(held_.front());
  held_.pop_front();
  if (speech) {
    forwarded_end_ = stacked_frames_ - static_cast<long>(held_.size());
    Pass(std::move(frame), out);
  }
}

void Frontend::EndSegment(std::vector<std::vector<float>>* out) {
  // greco's SubsampleStream restarts its count at the utterance end.
  passed_ = 0;
  Stack2Flush(out);
}

void Frontend::Pass(std::vector<float> frame, std::vector<std::vector<float>>* out) {
  if (!cfg_.subsample_after_gate) {
    Forward(std::move(frame), forwarded_end_, out);
    return;
  }
  if (passed_++ % cfg_.stack1.subsample != 0) return;
  // libsoda dates a kept frame by the last of the frames it stands for
  // (observed: word starts in ko-KR, hi-IN, pt-BR and cmn-Hans-CN match only
  // so).
  Forward(std::move(frame), forwarded_end_ + cfg_.stack1.subsample - 1, out);
}

void Frontend::Stack2Flush(std::vector<std::vector<float>>* out) {
  const size_t before = out->size();
  stack2_.Flush(out);
  if (output_ends_) output_ends_->insert(output_ends_->end(), out->size() - before, forwarded_end_);
}

void Frontend::ProcessFrame(const float* x, std::vector<std::vector<float>>* out) {
  // greco's framer passes frame_len_ + 1 samples. Its Window (with
  // htk_preemphasis_compatibility, the default) windows the first frame_len_,
  // pre-emphasizing the first against itself as (1 - p) x[0]; the extra
  // sample only counts towards the dither index.
  float sum_abs = 0.0f;
  for (int i = 0; i <= frame_len_; ++i) sum_abs += std::fabs(x[i]);
  // The start index also advances with the frame number, so even all-zero
  // frames get different dither.
  unsigned noise_index =
      (static_cast<unsigned>(sum_abs) % kNoiseTableSize + kNoiseFrameStride * frame_index_) %
      kNoiseTableSize;
  ++frame_index_;

  if (power_tap_ && tap_noise_scale_ >= 0.0f) {
    Window(x, tap_noise_scale_, noise_index);
    fft_.Compute(windowed_, &tap_power_);
    power_tap_->push_back(tap_power_);
  }
  Window(x, cfg_.noise_scale, noise_index);
  fft_.Compute(windowed_, &power_);
  if (power_tap_ && tap_noise_scale_ < 0.0f) power_tap_->push_back(power_);
  mel_.Compute(power_, &mel_out_);
  // greco's FilterBank takes the double-precision log.
  for (float& v : mel_out_) {
    v = static_cast<float>(std::log(static_cast<double>(std::max(v, cfg_.fbank_output_floor))));
  }

  std::vector<std::vector<float>> stacked;
  stack1_.Push(mel_out_, &stacked);
  PostStack1(stacked, out);
}

void Frontend::Window(const float* x, float noise_scale, unsigned noise_index) {
  windowed_.resize(frame_len_);
  const bool dither = noise_scale > 0.0f;
  for (int k = 0; k < frame_len_; ++k) {
    const float noise = noise_scale * noise_[noise_index];
    noise_index = (noise_index + kNoiseSampleStride) % kNoiseTableSize;
    if (k > 0) {
      const float v = x[k] - cfg_.preemphasis * x[k - 1];
      windowed_[k] = window_[k] * (dither ? noise + v : v);
    } else if (dither) {
      // greco takes the first sample in double.
      windowed_[k] = static_cast<float>(((1.0 - cfg_.preemphasis) * x[0] + noise) * window_[k]);
    } else {
      windowed_[k] = static_cast<float>((1.0 - cfg_.preemphasis) * (window_[k] * x[0]));
    }
  }
}

void Frontend::PostStack1(std::vector<std::vector<float>>& stacked,
                          std::vector<std::vector<float>>* out) {
  for (auto& f : stacked) {
    if (stacked_tap_) stacked_tap_->push_back(f);
    ++stacked_frames_;
    if (gated_) {
      held_.push_back(std::move(f));
    } else {
      forwarded_end_ = stacked_frames_;
      Pass(std::move(f), out);
    }
  }
}

void Frontend::Forward(std::vector<float> f, long end, std::vector<std::vector<float>>* out) {
  // Like greco's FrameNormalizeStream, divide (multiplying by the inverse
  // rounds differently).
  for (size_t i = 0; i < f.size(); ++i) f[i] = (f[i] - mean_[i]) / stddev_[i];
  if (cfg_.cluster_id >= 0) f.push_back(static_cast<float>(cfg_.cluster_id));
  const size_t before = out->size();
  stack2_.Push(f, out);
  if (output_ends_) output_ends_->insert(output_ends_->end(), out->size() - before, end);
}

}  // namespace soda

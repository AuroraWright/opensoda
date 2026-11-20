#include "vad_frontend.h"

#include <algorithm>
#include <cmath>

namespace soda {

void DataHistoryCdf::Init(int input_size, int max_history, int sample_every_n,
                          const std::vector<float>& percentiles) {
  input_size_ = input_size;
  limit_ = (max_history & ~1) - 1;
  sample_every_n_ = std::max(sample_every_n, 1);
  percentiles_ = percentiles;
  sorted_.assign(input_size, {});
  for (auto& s : sorted_) s.reserve(max_history);
  estimates_.assign(percentiles.size(), std::vector<float>(input_size, 0.0f));
  Reset();
}

void DataHistoryCdf::Reset() {
  downsampled_ = false;
  count_ = 0;
  frames_ = 0;
  for (auto& s : sorted_) s.clear();
  for (auto& e : estimates_) std::fill(e.begin(), e.end(), 0.0f);
}

void DataHistoryCdf::Add(const std::vector<float>& frame) {
  if (count_ >= limit_) {
    // Keep the odd-ranked half of each history.
    downsampled_ = true;
    count_ = limit_ / 2;
    for (int d = 0; d < input_size_; ++d) {
      std::vector<float>& s = sorted_[d];
      for (int k = 0; k < count_; ++k) s[k] = s[2 * k + 1];
      s.resize(count_);
      Update(d);
    }
  }
  if (!downsampled_ || frames_ % sample_every_n_ == 0) {
    for (int d = 0; d < input_size_; ++d) {
      std::vector<float>& s = sorted_[d];
      s.insert(std::upper_bound(s.begin(), s.end(), frame[d]), frame[d]);
      Update(d);
    }
    ++count_;
  }
  ++frames_;
}

void DataHistoryCdf::Update(int dim) {
  const std::vector<float>& v = sorted_[dim];
  const int n = static_cast<int>(v.size());
  for (size_t i = 0; i < percentiles_.size(); ++i) {
    const float p = percentiles_[i];
    float value;
    if (n == 0) {
      value = 0.0f;
    } else if (n == 1) {
      value = v[0];
    } else {
      // The sorted values sit at percentiles step, 2 step, ... n step;
      // beyond them the end slopes are extended.
      const float step = 1.0f / static_cast<float>(n + 1);
      if (p <= step) {
        value = v[0] - (1.0f - p / step) * (v[1] - v[0]);
      } else if (1.0 - static_cast<double>(step) <= static_cast<double>(p)) {
        const float frac = static_cast<float>(
            (static_cast<double>(p) - (1.0 - static_cast<double>(step))) / static_cast<double>(step));
        value = v[n - 1] + (v[n - 1] - v[n - 2]) * frac;
      } else {
        const float x = (p - step) / step;
        const int k = static_cast<int>(std::floor(x));
        const float frac = x - static_cast<float>(k);
        const float lo = v[std::min(k, n - 1)];
        const float hi = v[std::min(k + 1, n - 1)];
        value = lo + (hi - lo) * frac;
      }
    }
    estimates_[i][dim] = value;
  }
}

void VadFrontend::Init(const VadConfig& config, int num_bins, double sample_rate) {
  config_ = config;
  num_bins_ = num_bins;
  cdf_.Init(num_bins, config.noise.num_frames_start_downsampling,
            config.noise.sample_every_n_frames, config.noise.percentiles);
  mel_.Init(num_bins, sample_rate, config.channel_count, config.lower_band_limit,
            config.upper_band_limit);
  Reset();
}

void VadFrontend::Reset() {
  cdf_.Reset();
  frames_ = 0;
  started_ = false;
  buffered_.clear();
}

void VadFrontend::Push(const std::vector<float>& power, std::vector<std::vector<float>>* out) {
  ++frames_;
  cdf_.Add(power);
  if (started_) {
    Emit(power, out);
    return;
  }
  buffered_.push_back(power);
  if (frames_ >= config_.noise.initial_buffer_num_frames) Flush(out);
}

void VadFrontend::Flush(std::vector<std::vector<float>>* out) {
  for (const auto& f : buffered_) Emit(f, out);
  buffered_.clear();
  started_ = true;
}

void VadFrontend::Emit(const std::vector<float>& power, std::vector<std::vector<float>>* out) {
  // SpectralNoise's output: the input, then a noise level per percentile.
  std::vector<const float*> vectors;
  if (config_.noise.add_input_to_output) vectors.push_back(power.data());
  for (size_t i = 0; i < cdf_.num_percentiles(); ++i) vectors.push_back(cdf_.estimate(i).data());
  if (static_cast<int>(vectors.size()) < 3) return;

  // SpectralSubtractor: subtracts the floor noise magnitude, scaled down for
  // quiet noise and for a high peak-to-floor ratio, keeping at least
  // beta_floor_scale of the magnitude.
  const SpectralSubtractorConfig& c = config_.subtractor;
  subtracted_.resize(num_bins_);
  for (int k = 0; k < num_bins_; ++k) {
    const float x = std::max(vectors[0][k], 0.0f);
    const float peak = std::max(vectors[1][k], 1e-6f);
    const float floor = std::max(vectors[2][k], 1e-6f);
    const float noise = std::sqrt(floor);
    const float snr = std::log10(peak / floor) * 10.0f;
    float alpha = 0.0f;
    if (noise >= c.noise_low_subtraction_threshold) {
      float over = c.oversubtraction_factor;
      if (noise < c.noise_high_subtraction_threshold) {
        over *= (noise - c.noise_low_subtraction_threshold) *
                (1.0f / std::fmax(c.noise_high_subtraction_threshold -
                                      c.noise_low_subtraction_threshold,
                                  0.001f));
      }
      float scale;
      if (snr < c.snr_low_subtraction_threshold) {
        scale = c.snr_low_scale;
      } else if (snr <= c.snr_high_subtraction_threshold) {
        const float t = (snr - c.snr_low_subtraction_threshold) *
                        (1.0f / (c.snr_high_subtraction_threshold -
                                 c.snr_low_subtraction_threshold));
        scale = (1.0f - t) * c.snr_low_scale + t * c.snr_high_scale;
      } else {
        scale = c.snr_high_scale;
      }
      alpha = over * scale;
    }
    const float magnitude = std::sqrt(x);
    const float v = std::max(magnitude - noise * alpha, magnitude * c.beta_floor_scale);
    subtracted_[k] = v * v;
  }

  mel_.Compute(subtracted_, &mel_out_);
  // Like the main FilterBank, the double-precision log.
  for (float& v : mel_out_) {
    v = static_cast<float>(std::log(static_cast<double>(std::max(v, config_.fbank_output_floor))));
  }
  out->push_back(mel_out_);
}

}  // namespace soda

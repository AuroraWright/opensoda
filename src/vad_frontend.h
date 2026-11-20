// The noise-robust VAD features of ko-KR's and hi-IN's packs, following
// greco's "vad.*" streams:
//
//   FFT energies -> SpectralNoise (per-bin noise percentiles)
//   -> SpectralSubtractor (spectral subtraction) -> FilterBank (log mel)
//
// SpectralNoise holds back its first frames until it has seen
// initial_buffer_num_frames of them, so they too get a noise estimate.

#ifndef SODA_VAD_FRONTEND_H_
#define SODA_VAD_FRONTEND_H_

#include <deque>
#include <vector>

#include "config.h"
#include "frontend.h"

namespace soda {

// greco's DataHistoryCDF: for each input dimension, a sorted history of the
// values seen, read out at fixed percentiles. Once the history reaches
// num_frames_start_downsampling - 1 values, every other one is dropped and
// only every sample_every_n-th frame is added from then on.
class DataHistoryCdf {
 public:
  void Init(int input_size, int max_history, int sample_every_n,
            const std::vector<float>& percentiles);
  void Reset();
  void Add(const std::vector<float>& frame);
  // The value at percentile p for each input dimension.
  const std::vector<float>& estimate(size_t p) const { return estimates_[p]; }
  size_t num_percentiles() const { return percentiles_.size(); }

 private:
  void Update(int dim);

  int input_size_ = 0;
  int limit_ = 0;
  int sample_every_n_ = 1;
  std::vector<float> percentiles_;
  bool downsampled_ = false;
  int count_ = 0;   // Values in each history.
  long frames_ = 0;  // Frames offered since the last reset.
  std::vector<std::vector<float>> sorted_;
  std::vector<std::vector<float>> estimates_;
};

class VadFrontend {
 public:
  void Init(const VadConfig& config, int num_bins, double sample_rate);
  void Reset();
  // Takes one frame of FFT energies; appends the feature frames that are
  // ready.
  void Push(const std::vector<float>& power, std::vector<std::vector<float>>* out);
  // Emits the frames still held back (greco does so at the utterance end).
  void Flush(std::vector<std::vector<float>>* out);

 private:
  void Emit(const std::vector<float>& power, std::vector<std::vector<float>>* out);

  VadConfig config_;
  int num_bins_ = 0;
  DataHistoryCdf cdf_;
  MelFilterbank mel_;
  int frames_ = 0;
  bool started_ = false;  // The initial buffer has been released.
  std::deque<std::vector<float>> buffered_;
  std::vector<float> subtracted_, mel_out_;
};

}  // namespace soda

#endif  // SODA_VAD_FRONTEND_H_

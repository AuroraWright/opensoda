#include "config.h"

#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string_view>
#include <vector>

#include "proto_wire.h"

namespace soda {
namespace {

// Extension field numbers of greco.Params (and nested extension points), as
// declared in the descriptors embedded in libsoda.
constexpr uint32_t kFramerParams = 18905525;
constexpr uint32_t kWindowParams = 18905515;
constexpr uint32_t kFftParams = 18905514;
constexpr uint32_t kFilterBankParams = 18905716;
constexpr uint32_t kFrameStackerParams = 19077915;
constexpr uint32_t kSubsampleParams = 3328023;
constexpr uint32_t kAppendClusterIdParams = 247260833;
constexpr uint32_t kFrameNormalizeParams = 332044632;
constexpr uint32_t kEncoderStreamParams = 320100746;
constexpr uint32_t kRightContextParams = 39816938;
constexpr uint32_t kDomainIdMapping = 98179;
constexpr uint32_t kTfLiteSessionParams = 307419518;
constexpr uint32_t kProtoParams = 23431063;
constexpr uint32_t kEndpointerParams = 19583778;
constexpr uint32_t kLstmComputeParams = 84393869;
constexpr uint32_t kNeuralNetworkParams = 94228187;
constexpr uint32_t kSegmenterParams = 27058394;
constexpr uint32_t kSpectralNoiseParams = 26169916;
constexpr uint32_t kSpectralSubtractorParams = 27202700;
constexpr uint32_t kSessionRuntimeParams = 399795227;
constexpr uint32_t kDualSearchParams = 171805764;  // In FstSearchParams.
constexpr uint32_t kCapitalizationParams = 331244104;  // CapitalizationNormalizerParams
constexpr uint32_t kFstNormalizerParams = 27845124;
constexpr uint32_t kPunctuationNormalizerParams = 211152396;
constexpr uint32_t kPqrnnSpokenPunctuationParams = 387838176;
constexpr uint32_t kDenormalizerResourceList = 329822714;
constexpr int kPornDenormalizer = 1;  // DenormalizerResourceRef.denormalizer_type
constexpr uint32_t kSymbolTableParams = 98173;
constexpr uint32_t kAutoGainControlParams = 27582545;
constexpr uint32_t kTfliteRunStreamParams = 2441123;
constexpr uint32_t kLangIdParams = 62753032;
constexpr uint32_t kDecoderParams = 19583231;
constexpr uint32_t kFstDecoderParams = 41039125;            // In DecoderParams.
constexpr uint32_t kPieRnnFstDecoderParams = 194955910;     // In DecoderParams.
constexpr uint32_t kTfliteTransducerParams = 213412442;     // In PieRnnFst...
constexpr uint32_t kTfLiteModelResourceParams = 234621089;
constexpr uint32_t kSpeakerTurnStreamParams = 301915449;

struct Def {
  std::string name;
  std::string classname;
  std::string params;  // Serialized greco.Params.
  std::vector<std::string> params_refs;
};

Def ParseDef(std::string_view data) {
  Def d;
  ProtoReader r(data);
  while (r.Next()) {
    switch (r.field()) {
      case 1: d.name = r.str(); break;
      case 2: d.classname = r.str(); break;
      case 3: d.params = r.str(); break;
      case 4: d.params_refs.push_back(r.str()); break;
    }
  }
  return d;
}

// Returns the payload of extension `ext` inside a serialized message, or an
// empty view if absent.
std::string_view FindExtension(std::string_view msg, uint32_t ext) {
  ProtoReader r(msg);
  while (r.Next()) {
    if (r.field() == ext && r.wire_type() == kLengthDelimited) return r.bytes();
  }
  return {};
}

std::string FirstString(std::string_view msg, uint32_t field) {
  ProtoReader r(msg);
  while (r.Next()) {
    if (r.field() == field && r.wire_type() == kLengthDelimited) return r.str();
  }
  return {};
}

void ParseFramer(std::string_view m, FrontendConfig* c) {
  ProtoReader r(m);
  while (r.Next()) {
    if (r.field() == 1) c->frame_size_ms = r.fixed_float();
    if (r.field() == 2) c->frame_step_ms = r.fixed_float();
  }
}

void ParseWindow(std::string_view m, FrontendConfig* c) {
  ProtoReader r(m);
  while (r.Next()) {
    switch (r.field()) {
      case 1: c->preemphasis = r.fixed_float(); break;
      case 3: c->window_type = static_cast<int>(r.varint()); break;
      case 6: c->noise_scale = r.fixed_float(); break;
      case 8: c->window_size_ms = r.fixed_float(); break;
    }
  }
}

void ParseFft(std::string_view m, FrontendConfig* c) {
  ProtoReader r(m);
  while (r.Next()) {
    switch (r.field()) {
      case 1: c->fft_input_length_ms = static_cast<int>(r.varint()); break;
      case 2: c->min_fft_size = static_cast<int>(r.varint()); break;
      case 5: c->fft_overdrive = r.boolean(); break;
    }
  }
}

void ParseFilterBank(std::string_view m, FrontendConfig* c) {
  ProtoReader r(m);
  while (r.Next()) {
    switch (r.field()) {
      case 1: c->channel_count = static_cast<int>(r.varint()); break;
      case 2: c->upper_band_limit = r.fixed_float(); break;
      case 3: c->lower_band_limit = r.fixed_float(); break;
      case 7: c->fbank_output_floor = r.fixed_float(); break;
    }
  }
}

void ParseSpectralNoise(std::string_view m, SpectralNoiseConfig* c) {
  float peak = 0.95f, median = 0.5f, background = 0.3f, floor = 0.25f;
  bool use_peak = true, use_median = false, use_background = false, use_floor = true;
  ProtoReader r(m);
  while (r.Next()) {
    switch (r.field()) {
      case 3: c->initial_buffer_num_frames = static_cast<int>(r.int64()); break;
      case 4: c->num_frames_start_downsampling = static_cast<int>(r.int64()); break;
      case 5: c->sample_every_n_frames = static_cast<int>(r.int64()); break;
      case 8: floor = r.fixed_float(); break;
      case 9: background = r.fixed_float(); break;
      case 10: median = r.fixed_float(); break;
      case 11: peak = r.fixed_float(); break;
      case 12: use_peak = r.boolean(); break;
      case 13: use_median = r.boolean(); break;
      case 14: use_background = r.boolean(); break;
      case 15: use_floor = r.boolean(); break;
      case 16: c->add_input_to_output = r.boolean(); break;
    }
  }
  c->percentiles.clear();
  if (use_peak) c->percentiles.push_back(peak);
  if (use_median) c->percentiles.push_back(median);
  if (use_background) c->percentiles.push_back(background);
  if (use_floor) c->percentiles.push_back(floor);
}

void ParseSpectralSubtractor(std::string_view m, SpectralSubtractorConfig* c) {
  ProtoReader r(m);
  while (r.Next()) {
    switch (r.field()) {
      case 3: c->number_spectral_vectors = static_cast<int>(r.int64()); break;
      case 4: c->oversubtraction_factor = r.fixed_float(); break;
      case 5: c->noise_low_subtraction_threshold = r.fixed_float(); break;
      case 6: c->noise_high_subtraction_threshold = r.fixed_float(); break;
      case 7: c->beta_floor_scale = r.fixed_float(); break;
      case 8: c->snr_low_subtraction_threshold = r.fixed_float(); break;
      case 9: c->snr_high_subtraction_threshold = r.fixed_float(); break;
      case 10: c->snr_low_scale = r.fixed_float(); break;
      case 11: c->snr_high_scale = r.fixed_float(); break;
    }
  }
}

void ParseFrameStacker(std::string_view m, FrameStackConfig* s) {
  // Proto defaults: left_context = 4, right_context = 0, input_size = 13.
  s->left_context = 4;
  s->right_context = 0;
  s->input_size = 13;
  s->zero_padding = false;
  ProtoReader r(m);
  while (r.Next()) {
    switch (r.field()) {
      case 2: s->input_size = static_cast<int>(r.varint()); break;
      case 3: s->left_context = static_cast<int>(r.varint()); break;
      case 4: s->right_context = static_cast<int>(r.varint()); break;
      case 7: s->zero_padding = r.boolean(); break;
    }
  }
}

int ParseSubsample(std::string_view m) {
  int stride = 1;
  ProtoReader r(m);
  while (r.Next()) {
    if (r.field() == 1) stride = static_cast<int>(r.varint());
  }
  return stride;
}

void ParseDecoderParams(std::string_view m, DecoderConfig* d,
                        std::map<std::string, std::string>* session_refs) {
  std::string_view fst = FindExtension(m, kFstDecoderParams);
  if (!fst.empty()) {
    (*session_refs)["search"] = FirstString(fst, 11);
    ProtoReader f(fst);
    while (f.Next()) {
      if (f.field() == 24) d->resume_search = f.boolean();  // enable_resume_search
      if (f.field() == 38) d->ignored_resume_context_suffix = f.str();
    }
  }
  std::string_view pie = FindExtension(m, kPieRnnFstDecoderParams);
  if (pie.empty()) return;
  ProtoReader r(pie);
  while (r.Next()) {
    switch (r.field()) {
      case 2: d->sos_id = static_cast<int>(r.int64()); break;
      case 11: d->eos_id = static_cast<int>(r.int64()); break;
      case 4: d->smoothing_temperature = r.fixed_float(); break;
      case 16: d->internal_lm_smoothing_temperature = r.fixed_float(); break;
      case 13: d->eos_penalty = r.fixed_float(); break;
    }
  }
  std::string_view tt = FindExtension(pie, kTfliteTransducerParams);
  d->subtract_internal_lm = true;  // Proto default.
  d->internal_lm_cost_scale = 1.0f;
  ProtoReader t(tt);
  while (t.Next()) {
    switch (t.field()) {
      case 5: (*session_refs)["decoder"] = t.str(); break;
      case 6: (*session_refs)["joint"] = t.str(); break;
      case 16: (*session_refs)["joint_prior"] = t.str(); break;
      case 9: d->is_hat_model = t.boolean(); break;
      case 10: d->internal_lm_cost_scale = t.fixed_float(); break;
      case 11: d->subtract_internal_lm = t.boolean(); break;
      case 12:
        if (t.wire_type() == kVarint) {
          d->decoder_initial_value = static_cast<int>(t.int64());
        }
        break;
    }
  }
}

void ParseSearchParams(std::string_view m, DecoderConfig* d) {
  ProtoReader r(m);
  while (r.Next()) {
    switch (r.field()) {
      case 1: d->max_arcs = static_cast<int>(r.varint()); break;
      case 2: d->local_beam = r.fixed_float(); break;
      case 3: d->histogram_buckets = static_cast<int>(r.varint()); break;
      case 7: d->word_pen = r.fixed_float(); break;
      case 20: d->histogram_scale = r.fixed_float(); break;
      case 23: d->min_arcs = static_cast<int>(r.varint()); break;
    }
  }
  // DualSearchParams: {1: max_model_cost, 9: max_expansions,
  // 33: use_min_of_original_and_influenced_cost_for_pruning,
  // 34: max_model_cost_to_influence}.
  ProtoReader dual(FindExtension(m, kDualSearchParams));
  while (dual.Next()) {
    if (dual.field() == 1) d->max_model_cost = dual.fixed_float();
    if (dual.field() == 9) d->max_expansions = static_cast<int>(dual.varint());
    if (dual.field() == 33) d->prune_on_min_cost = dual.boolean();
    if (dual.field() == 34) d->max_model_cost_to_influence = dual.fixed_float();
  }
}

void ParseEndpointer(std::string_view msg, EndpointerConfig* e) {
  ProtoReader r(msg);
  while (r.Next()) {
    switch (r.field()) {
      case 1: e->posterior_dimension = static_cast<int>(r.int64()); break;
      case 2: e->frame_period = r.fixed_float(); break;
      case 3: e->endpoint_margin = r.fixed_float(); break;
      case 4: e->onset_window = r.fixed_float(); break;
      case 5: e->speech_on_window = r.fixed_float(); break;
      case 6: e->offset_window = r.fixed_float(); break;
      case 7: e->onset_detect_dur = r.fixed_float(); break;
      case 8: e->onset_confirm_dur = r.fixed_float(); break;
      case 9: e->on_maintain_dur = r.fixed_float(); break;
      case 10: e->offset_confirm_dur = r.fixed_float(); break;
      case 11: e->decision_threshold = r.fixed_float(); break;
      case 14: e->never_transition_to_silence = r.varint() != 0; break;
      case 21: e->additional_margin_before_speech = r.fixed_float(); break;
      case 22: e->additional_margin_after_speech = r.fixed_float(); break;
    }
  }
}

void ParseAgc(std::string_view m, AgcConfig* a) {
  a->activated = true;
  ProtoReader r(m);
  while (r.Next()) {
    switch (r.field()) {
      case 1: a->delay_ms = static_cast<int>(r.int64()); break;
      case 2: a->half_life_sec = r.fixed_float(); break;
      case 4: a->floor = r.fixed_float(); break;
      case 5: a->curvature = r.fixed_float(); break;
      case 6: a->attack_window_ms = static_cast<int>(r.int64()); break;
      case 7: a->estimator = static_cast<int>(r.int64()); break;
      case 8: a->level_target = r.fixed_float(); break;
      case 9: a->lower_level_target = r.fixed_float(); break;
      case 10: a->activated = r.boolean(); break;
      case 11: a->gain_scale = r.fixed_float(); break;
      case 13: a->lower_target_transition = r.fixed_float(); break;
    }
  }
}

void ParseConfidence(std::string_view m, LangIdConfidenceConfig* c) {
  bool confident = false, highly = false;
  ProtoReader r(m);
  while (r.Next()) {
    switch (r.field()) {
      case 1: c->prob_weight = r.fixed_float(); break;
      case 2: c->prob_delta_weight = r.fixed_float(); break;
      case 3: c->negentropy_weight = r.fixed_float(); break;
      case 4: c->continuity_weight = r.fixed_float(); break;
      case 5: c->confident_threshold = r.fixed_float(); confident = true; break;
      case 6: c->highly_confident_threshold = r.fixed_float(); highly = true; break;
      case 7: c->continuity_upper_bound = r.fixed_float(); break;
    }
  }
  c->has_thresholds = confident && highly;
}

// Reads a PipelineDef's streams and resources (keyed by name).
bool ReadPipeline(const std::string& path, std::vector<Def>* streams,
                  std::map<std::string, Def>* resources, std::string* error) {
  std::string data;
  if (!ReadFile(path, &data)) {
    *error = "cannot read " + path;
    return false;
  }
  ProtoReader r(data);
  while (r.Next()) {
    if (r.field() == 1) {
      streams->push_back(ParseDef(r.bytes()));
    } else if (r.field() == 3 || r.field() == 5) {
      Def d = ParseDef(r.bytes());
      (*resources)[d.name] = d;
    }
  }
  if (!r.ok() || streams->empty()) {
    *error = "malformed pipeline config " + path;
    return false;
  }
  return true;
}

// LstmComputeStreamParams: the model and normalization refs, and the
// periodic state reset.
void ParseLstmCompute(std::string_view lp, VadConfig* vad, std::string* model_ref,
                      std::string* mean_stddev_ref) {
  *model_ref = FirstString(lp, 11);
  *mean_stddev_ref = FirstString(lp, 13);
  bool reset_state = true;
  int min_frames = 0;
  ProtoReader r(lp);
  while (r.Next()) {
    switch (r.field()) {
      case 17: reset_state = r.boolean(); break;
      case 18: min_frames = static_cast<int>(r.int64()); break;
      case 19: vad->reset_frames_below = static_cast<int>(r.int64()); break;
      case 21: vad->reset_threshold = r.fixed_float(); break;
    }
  }
  vad->min_frames_before_reset = reset_state ? min_frames : 0;
}

}  // namespace

bool LoadLangIdConfig(const std::string& path, LangIdConfig* config,
                      std::string* error) {
  std::vector<Def> streams;
  std::map<std::string, Def> resources;
  if (!ReadPipeline(path, &streams, &resources, error)) return false;
  auto resource_params = [&](const std::string& ref) -> std::string_view {
    auto it = resources.find(ref);
    return it == resources.end() ? std::string_view() : it->second.params;
  };
  FrontendConfig& fe = config->frontend;
  fe.stack2 = {0, 0, 0, false, 1};
  fe.cluster_id = -1;
  std::string vad_model_ref, vad_mean_stddev_ref, session_ref, langid_params_ref;
  for (const Def& s : streams) {
    std::vector<std::string_view> params{s.params};
    for (const auto& ref : s.params_refs) params.push_back(resource_params(ref));
    for (std::string_view p : params) {
      if (s.classname == "Framer") {
        ParseFramer(FindExtension(p, kFramerParams), &fe);
      } else if (s.classname == "Window") {
        ParseWindow(FindExtension(p, kWindowParams), &fe);
      } else if (s.classname == "Fft") {
        ParseFft(FindExtension(p, kFftParams), &fe);
      } else if (s.classname == "FilterBank") {
        if (auto fb = FindExtension(p, kFilterBankParams); !fb.empty()) ParseFilterBank(fb, &fe);
        if (auto fft = FindExtension(p, kFftParams); !fft.empty()) ParseFft(fft, &fe);
      } else if (s.classname == "LangIdStream") {
        std::string_view lp = FindExtension(p, kLangIdParams);
        if (std::string ref = FirstString(lp, 8); !ref.empty()) langid_params_ref = ref;
        ProtoReader r(lp);
        while (r.Next()) {
          if (r.field() == 9) ParseConfidence(r.bytes(), &config->confidence);
        }
      }
    }
    if (s.classname == "AutoGainControl") {
      ParseAgc(FindExtension(s.params, kAutoGainControlParams), &config->agc);
    } else if (s.classname == "FrameStacker") {
      ParseFrameStacker(FindExtension(s.params, kFrameStackerParams), &fe.stack1);
      fe.stack1.subsample = 1;
    } else if (s.classname == "SubsampleStream") {
      fe.stack1.subsample = ParseSubsample(FindExtension(s.params, kSubsampleParams));
    } else if (s.classname == "AppendClusterIdStream") {
      ProtoReader c(FindExtension(s.params, kAppendClusterIdParams));
      bool one_hot = false;
      int clusters = 0;
      while (c.Next()) {
        if (c.field() == 1) config->vad.cluster_id = static_cast<int>(c.int64());
        if (c.field() == 2) one_hot = c.boolean();
        if (c.field() == 3) clusters = static_cast<int>(c.int64());
      }
      config->vad.num_clusters = one_hot ? clusters : 0;
    } else if (s.classname == "LstmComputeStream") {
      ParseLstmCompute(FindExtension(s.params, kLstmComputeParams), &config->vad,
                       &vad_model_ref, &vad_mean_stddev_ref);
    } else if (s.classname == "EndpointerStream") {
      ParseEndpointer(FindExtension(s.params, kEndpointerParams), &config->vad.endpointer);
    } else if (s.classname == "SegmenterStream") {
      ProtoReader g(FindExtension(s.params, kSegmenterParams));
      while (g.Next()) {
        if (g.field() == 3) config->reset_interval_ms = static_cast<int>(g.int64());
      }
    } else if (s.classname == "TfLiteSessionRunStream") {
      ProtoReader r(FindExtension(s.params, kTfliteRunStreamParams));
      while (r.Next()) {
        switch (r.field()) {
          case 12: config->frame_batch_size = static_cast<int>(r.int64()); break;
          case 18: config->reset_segment_state = r.boolean(); break;
          case 19: config->reset_every_n_frame = static_cast<int>(r.int64()); break;
          case 20: session_ref = r.str(); break;
        }
      }
    }
  }
  auto filename = [&](const std::string& ref, uint32_t ext) {
    return FirstString(FindExtension(resource_params(ref), ext), 1);
  };
  config->model_file = filename(session_ref, kTfLiteSessionParams);
  config->application_params_file = filename(langid_params_ref, kProtoParams);
  config->vad.model_file = filename(vad_model_ref, kNeuralNetworkParams);
  config->vad.mean_stddev_file = filename(vad_mean_stddev_ref, kProtoParams);
  fe.stack2.input_size = fe.stack1.input_size * (fe.stack1.left_context + 1 + fe.stack1.right_context);
  if (config->model_file.empty() || config->application_params_file.empty() ||
      config->vad.model_file.empty()) {
    *error = "language identification config lacks its model files";
    return false;
  }
  return true;
}

bool LoadSpeakerTurnConfig(const std::string& path, SpeakerTurnConfig* config,
                           std::string* error) {
  std::vector<Def> streams;
  std::map<std::string, Def> resources;
  if (!ReadPipeline(path, &streams, &resources, error)) return false;
  auto resource_params = [&](const std::string& ref) -> std::string_view {
    auto it = resources.find(ref);
    return it == resources.end() ? std::string_view() : it->second.params;
  };
  auto filename = [&](const std::string& ref, uint32_t ext) {
    return FirstString(FindExtension(resource_params(ref), ext), 1);
  };
  FrontendConfig& fe = config->frontend;
  fe.stack2 = {0, 0, 0, false, 1};
  fe.cluster_id = -1;
  std::map<std::string, std::string> refs;
  std::string encoder_ref, mean_stddev_ref, vad_model_ref, vad_mean_stddev_ref;
  for (const Def& s : streams) {
    std::vector<std::string_view> params{s.params};
    for (const auto& ref : s.params_refs) params.push_back(resource_params(ref));
    for (std::string_view p : params) {
      if (s.classname == "Framer") {
        ParseFramer(FindExtension(p, kFramerParams), &fe);
      } else if (s.classname == "Window") {
        ParseWindow(FindExtension(p, kWindowParams), &fe);
      } else if (s.classname == "Fft") {
        ParseFft(FindExtension(p, kFftParams), &fe);
      } else if (s.classname == "FilterBank") {
        if (auto fb = FindExtension(p, kFilterBankParams); !fb.empty()) ParseFilterBank(fb, &fe);
        if (auto fft = FindExtension(p, kFftParams); !fft.empty()) ParseFft(fft, &fe);
      }
    }
    if (s.classname == "AutoGainControl") {
      ParseAgc(FindExtension(s.params, kAutoGainControlParams), &config->agc);
    } else if (s.classname == "FrameStacker") {
      ParseFrameStacker(FindExtension(s.params, kFrameStackerParams), &fe.stack1);
      fe.stack1.subsample = 1;
    } else if (s.classname == "SubsampleStream") {
      fe.stack1.subsample = ParseSubsample(FindExtension(s.params, kSubsampleParams));
    } else if (s.classname == "AppendClusterIdStream") {
      ProtoReader c(FindExtension(s.params, kAppendClusterIdParams));
      bool one_hot = false;
      int clusters = 0;
      while (c.Next()) {
        if (c.field() == 1) config->vad.cluster_id = static_cast<int>(c.int64());
        if (c.field() == 2) one_hot = c.boolean();
        if (c.field() == 3) clusters = static_cast<int>(c.int64());
      }
      config->vad.num_clusters = one_hot ? clusters : 0;
    } else if (s.classname == "LstmComputeStream") {
      ParseLstmCompute(FindExtension(s.params, kLstmComputeParams), &config->vad,
                       &vad_model_ref, &vad_mean_stddev_ref);
    } else if (s.classname == "EndpointerStream") {
      ParseEndpointer(FindExtension(s.params, kEndpointerParams), &config->vad.endpointer);
    } else if (s.classname == "SegmenterStream") {
      ProtoReader g(FindExtension(s.params, kSegmenterParams));
      while (g.Next()) {
        if (g.field() == 3) config->reset_interval_ms = static_cast<int>(g.int64());
        if (g.field() == 4) config->vad.impatience_start_ms = static_cast<int>(g.int64());
        if (g.field() == 5) config->vad.impatience_limit_ms = static_cast<int>(g.int64());
      }
    } else if (s.classname == "TFliteRunStream") {
      ProtoReader r(FindExtension(s.params, kTfliteRunStreamParams));
      while (r.Next()) {
        switch (r.field()) {
          case 1: encoder_ref = r.str(); break;
          case 11: config->initial_activations_ignored = static_cast<int>(r.int64()); break;
          case 12: config->frame_batch_size = static_cast<int>(r.int64()); break;
          case 15: mean_stddev_ref = r.str(); break;
          case 16: config->last_frame_repetitions = static_cast<int>(r.int64()); break;
        }
      }
    } else if (s.classname == "Decoder2Stream") {
      std::string_view dp = FindExtension(s.params, kDecoderParams);
      ParseDecoderParams(dp, &config->decoder, &refs);
      refs["symbols"] = FirstString(dp, 3);
      // FstDecoderParams.search_params (10) is inline here.
      ProtoReader f(FindExtension(dp, kFstDecoderParams));
      while (f.Next()) {
        if (f.field() == 10) ParseSearchParams(f.bytes(), &config->decoder);
      }
    } else if (s.classname == "SpeakerTurnStream") {
      ProtoReader r(FindExtension(s.params, kSpeakerTurnStreamParams));
      while (r.Next()) {
        switch (r.field()) {
          case 1: config->speaker_change_id = static_cast<int>(r.int64()); break;
          case 2: config->time_offset_ms = static_cast<int>(r.int64()); break;
          case 3: config->max_speaker_segment_ms = static_cast<int>(r.int64()); break;
          case 4: config->context_labels = static_cast<int>(r.int64()); break;
          case 5: config->short_responses_to_ignore.push_back(r.str()); break;
        }
      }
    }
  }
  config->vad.model_file = filename(vad_model_ref, kNeuralNetworkParams);
  config->vad.mean_stddev_file = filename(vad_mean_stddev_ref, kProtoParams);
  config->encoder_file = filename(encoder_ref, kTfLiteModelResourceParams);
  config->mean_stddev_file = filename(mean_stddev_ref, kProtoParams);
  config->decoder_file = filename(refs["decoder"], kTfLiteSessionParams);
  config->joint_file = filename(refs["joint"], kTfLiteSessionParams);
  config->symbols_file = filename(refs["symbols"], kSymbolTableParams);
  if (config->encoder_file.empty() || config->mean_stddev_file.empty() ||
      config->decoder_file.empty() || config->joint_file.empty() ||
      config->symbols_file.empty() || config->speaker_change_id <= 0) {
    *error = "speaker turn config " + path + " lacks its models";
    return false;
  }
  return true;
}

bool ReadFile(const std::string& path, std::string* out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  std::ostringstream ss;
  ss << f.rdbuf();
  *out = ss.str();
  return true;
}

bool LoadMeanStdDev(const std::string& path, std::vector<float>* mean,
                    std::vector<float>* stddev, std::string* error) {
  std::string data;
  if (!ReadFile(path, &data)) {
    *error = "cannot read " + path;
    return false;
  }
  ProtoReader r(data);  // lstm.MeanStdDev {repeated float mean=1, std_dev=2}
  while (r.Next()) {
    if (r.field() == 1) r.AppendFloats(mean);
    if (r.field() == 2) r.AppendFloats(stddev);
  }
  if (!r.ok() || mean->empty() || mean->size() != stddev->size()) {
    *error = "malformed mean/stddev file " + path;
    return false;
  }
  return true;
}

// Finds a greco.DynamicClassTwiddlerModelParams (whose wpm_ref, field 6,
// names a PortableWordPieceModel resource) anywhere inside `msg`, and the
// name of the context model holding it (field 1 of the nearest enclosing
// message that has one).
bool FindTwiddlerParams(std::string_view msg, const std::map<std::string, Def>& resources,
                        BiasingConfig* b, int depth = 0) {
  if (depth > 12) return false;
  std::string wpm_ref;
  ProtoReader r(msg);
  while (r.Next()) {
    if (r.field() == 6 && r.wire_type() == kLengthDelimited) wpm_ref = r.str();
  }
  if (!r.ok()) return false;
  auto it = resources.find(wpm_ref);
  if (it != resources.end() && it->second.classname == "PortableWordPieceModel") {
    ProtoReader p(it->second.params);  // The only extension: {1: filename}.
    while (p.Next()) {
      if (p.wire_type() == kLengthDelimited) b->wpm_file = FirstString(p.bytes(), 1);
    }
    b->respect_boost = true;
    ProtoReader m(msg);
    while (m.Next()) {
      if (m.field() == 17 && m.wire_type() == kFixed64) {
        uint64_t bits = m.varint();
        double w;
        std::memcpy(&w, &bits, sizeof(w));
        b->token_weight = static_cast<float>(w);
      }
      if (m.field() == 18) b->respect_boost = m.boolean();
      if (m.field() == 2 && m.wire_type() == kLengthDelimited) {
        // FstTwiddlerParams {1: FstTwiddlerBuilderParams {4: retwiddle_penalty}}.
        ProtoReader builder(FindExtension(m.bytes(), 1));
        while (builder.Next()) {
          if (builder.field() == 4) b->retwiddle_penalty = builder.fixed_float();
        }
      }
    }
    if (!b->wpm_file.empty() && b->token_weight > 0) return true;
    *b = BiasingConfig();  // Another model's word-piece reference.
  }
  ProtoReader sub(msg);
  while (sub.Next()) {
    if (sub.wire_type() == kLengthDelimited &&
        FindTwiddlerParams(sub.bytes(), resources, b, depth + 1)) {
      if (b->context_name.empty()) b->context_name = FirstString(msg, 1);
      return true;
    }
  }
  return false;
}

bool LoadPipelineConfig(const std::string& path, PipelineConfig* config,
                        std::string* error) {
  std::string data;
  if (!ReadFile(path, &data)) {
    *error = "cannot read " + path;
    return false;
  }
  std::vector<Def> streams;
  std::map<std::string, Def> resources;
  ProtoReader r(data);
  while (r.Next()) {
    if (r.field() == 1) {
      streams.push_back(ParseDef(r.bytes()));
    } else if (r.field() == 3 || r.field() == 5) {
      Def d = ParseDef(r.bytes());
      resources[d.name] = d;
    }
  }
  if (!r.ok() || streams.empty()) {
    *error = "malformed pipeline config " + path;
    return false;
  }

  FrontendConfig& fe = config->frontend;
  fe.cluster_id = -1;
  // The first decoder and encoder make the first pass; a second of each
  // (after an EncoderInsertRightContextStream) makes the cascaded pass.
  DecoderConfig& dec = config->decoder;
  std::map<std::string, std::string> refs, cascade_refs;
  std::string encoder_ref, cascade_encoder_ref, mean_stddev_ref, vad_model_ref,
      vad_mean_stddev_ref;
  std::string encoder_domains_ref, cascade_domains_ref;
  int stackers_seen = 0, encoders_seen = 0, decoders_seen = 0;
  bool after_normalize = false, segmenter_seen = false;
  auto parse_encoder = [](std::string_view p, EncoderConfig* e, std::string* session_ref,
                          std::string* domains_ref) {
    std::vector<int> batches, flush;
    ProtoReader r(p);
    while (r.Next()) {
      if (r.field() == 1) *session_ref = r.str();
      if (r.field() == 8) r.AppendInts(&batches);
      if (r.field() == 16) r.AppendInts(&flush);
      if (r.field() == 17) *domains_ref = r.str();
    }
    if (!batches.empty()) e->time_batch_sizes = batches;
    if (!flush.empty()) e->flush_batch_sizes = flush;
  };

  // Parameters may sit in the stream itself or in referenced Params resources.
  auto params_of = [&](const Def& s) {
    std::vector<std::string_view> out{s.params};
    for (const auto& ref : s.params_refs) {
      auto it = resources.find(ref);
      if (it != resources.end()) out.push_back(it->second.params);
    }
    return out;
  };

  for (const Def& s : streams) {
    // The VAD branch shares the classes but is not on the encoder path.
    if (s.name.rfind("vad.", 0) == 0) {
      VadConfig& vad = config->vad;
      if (s.classname == "AppendClusterIdStream") {
        ProtoReader c(FindExtension(s.params, kAppendClusterIdParams));
        bool one_hot = false;
        int clusters = 0;
        while (c.Next()) {
          if (c.field() == 1) vad.cluster_id = static_cast<int>(c.int64());
          if (c.field() == 2) one_hot = c.boolean();
          if (c.field() == 3) clusters = static_cast<int>(c.int64());
          if (c.field() == 4) {  // DomainNameToInternalIdEntry {1: key, 2: value}.
            ProtoReader e(c.bytes());
            std::string key;
            int value = 0;
            while (e.Next()) {
              if (e.field() == 1) key = e.str();
              if (e.field() == 2) value = static_cast<int>(e.int64());
            }
            if (!key.empty()) config->vad_cluster_id_by_domain[key] = value;
          }
        }
        vad.num_clusters = one_hot ? clusters : 0;
      } else if (s.classname == "SpectralNoise") {
        vad.spectral_noise = true;
        ParseSpectralNoise(FindExtension(s.params, kSpectralNoiseParams), &vad.noise);
      } else if (s.classname == "SpectralSubtractor") {
        ParseSpectralSubtractor(FindExtension(s.params, kSpectralSubtractorParams),
                                &vad.subtractor);
      } else if (s.classname == "Window") {
        FrontendConfig window;
        ParseWindow(FindExtension(s.params, kWindowParams), &window);
        vad.noise_scale = window.noise_scale;
      } else if (s.classname == "FilterBank") {
        // Its FftParams are the main path's (common_fft_params).
        ProtoReader r(FindExtension(s.params, kFilterBankParams));
        while (r.Next()) {
          switch (r.field()) {
            case 1: vad.channel_count = static_cast<int>(r.varint()); break;
            case 2: vad.upper_band_limit = r.fixed_float(); break;
            case 3: vad.lower_band_limit = r.fixed_float(); break;
            case 7: vad.fbank_output_floor = r.fixed_float(); break;
          }
        }
      } else if (s.classname == "LstmComputeStream") {
        ParseLstmCompute(FindExtension(s.params, kLstmComputeParams), &vad, &vad_model_ref,
                         &vad_mean_stddev_ref);
      } else if (s.name == "vad.decision_for_segmenter") {
        ParseEndpointer(FindExtension(s.params, kEndpointerParams), &vad.endpointer);
      } else if (s.classname == "AudioLevelEventStream") {
        vad.audio_level_events = true;
      }
      continue;
    }
    if (s.classname == "AutoGainControl") {
      ParseAgc(FindExtension(s.params, kAutoGainControlParams), &config->agc);
      continue;
    }
    for (std::string_view p : params_of(s)) {
      if (s.classname == "Framer") {
        ParseFramer(FindExtension(p, kFramerParams), &fe);
      } else if (s.classname == "Window") {
        ParseWindow(FindExtension(p, kWindowParams), &fe);
      } else if (s.classname == "Fft") {
        ParseFft(FindExtension(p, kFftParams), &fe);
      } else if (s.classname == "FilterBank") {
        auto fb = FindExtension(p, kFilterBankParams);
        if (!fb.empty()) ParseFilterBank(fb, &fe);
        auto fft = FindExtension(p, kFftParams);
        if (!fft.empty()) ParseFft(fft, &fe);
      }
    }
    if (s.classname == "SegmenterStream") {
      segmenter_seen = true;
      ProtoReader g(FindExtension(s.params, kSegmenterParams));
      while (g.Next()) {
        if (g.field() == 4) config->vad.impatience_start_ms = static_cast<int>(g.int64());
        if (g.field() == 5) config->vad.impatience_limit_ms = static_cast<int>(g.int64());
      }
    } else if (s.classname == "FrameStacker") {
      FrameStackConfig* st = stackers_seen++ == 0 ? &fe.stack1 : &fe.stack2;
      ParseFrameStacker(FindExtension(s.params, kFrameStackerParams), st);
      st->subsample = 1;
    } else if (s.classname == "SubsampleStream") {
      int stride = ParseSubsample(FindExtension(s.params, kSubsampleParams));
      (stackers_seen <= 1 ? fe.stack1 : fe.stack2).subsample = stride;
      if (stackers_seen <= 1 && segmenter_seen) fe.subsample_after_gate = true;
    } else if (s.classname == "FrameNormalizeStream") {
      mean_stddev_ref =
          FirstString(FindExtension(s.params, kFrameNormalizeParams), 1);
      after_normalize = true;
    } else if (s.classname == "AppendClusterIdStream" && after_normalize) {
      ProtoReader c(FindExtension(s.params, kAppendClusterIdParams));
      while (c.Next()) {
        if (c.field() == 1) fe.cluster_id = static_cast<int>(c.int64());
        if (c.field() == 4) {
          ProtoReader e(c.bytes());
          std::string key;
          int value = 0;
          while (e.Next()) {
            if (e.field() == 1) key = e.str();
            if (e.field() == 2) value = static_cast<int>(e.int64());
          }
          if (!key.empty()) config->frontend_cluster_id_by_domain[key] = value;
        }
      }
    } else if (s.classname == "CausalEncoderStream") {
      std::string_view p = FindExtension(s.params, kEncoderStreamParams);
      if (encoders_seen++ == 0) {
        parse_encoder(p, &config->encoder, &encoder_ref, &encoder_domains_ref);
      } else {
        parse_encoder(p, &config->cascade.encoder, &cascade_encoder_ref, &cascade_domains_ref);
      }
    } else if (s.classname == "EncoderInsertRightContextStream") {
      ProtoReader r(FindExtension(s.params, kRightContextParams));
      while (r.Next()) {
        if (r.field() == 1) config->cascade.right_context = static_cast<int>(r.int64());
        if (r.field() == 2) config->cascade.use_last_frame = r.boolean();
      }
    } else if (s.classname == "DecoderEndpointerStream" || s.classname == "DecoderStream") {
      std::string_view dp = FindExtension(s.params, kDecoderParams);
      if (decoders_seen++ == 0) {
        ParseDecoderParams(dp, &dec, &refs);
        if (std::string syms = FirstString(dp, 3); !syms.empty()) refs["symbols"] = syms;
      } else {
        ParseDecoderParams(dp, &config->cascade.decoder, &cascade_refs);
      }
    }
  }

  auto filename_of = [&](const std::string& ref, uint32_t ext) {
    auto it = resources.find(ref);
    if (it == resources.end()) return std::string();
    std::string_view p = FindExtension(it->second.params, ext);
    std::string name = FirstString(p, 1);
    if (name.empty() && ext == kTfLiteSessionParams) {
      // Newer configs name the file in a SessionResourceRuntimeParams resource
      // (TfLiteSessionParams.session_resource_runtime_params_ref = 14).
      auto rt = resources.find(FirstString(p, 14));
      if (rt != resources.end()) {
        name = FirstString(FindExtension(rt->second.params, kSessionRuntimeParams), 1);
      }
    }
    return name;
  };
  auto domain_ids = [&](const std::string& ref, EncoderConfig* e) {
    auto it = resources.find(ref);
    if (it == resources.end()) return;
    e->default_domain_id = 0;
    ProtoReader m(FindExtension(it->second.params, kDomainIdMapping));
    while (m.Next()) {
      if (m.field() == 2) e->default_domain_id = static_cast<int>(m.int64());
      if (m.field() != 1) continue;
      ProtoReader entry(m.bytes());  // {1: key, 2: value}
      std::string key;
      int value = 0;
      while (entry.Next()) {
        if (entry.field() == 1) key = entry.str();
        if (entry.field() == 2) value = static_cast<int>(entry.int64());
      }
      e->domain_ids[key] = value;
    }
  };
  config->encoder.file = filename_of(encoder_ref, kTfLiteSessionParams);
  domain_ids(encoder_domains_ref, &config->encoder);
  config->decoder_file = filename_of(refs["decoder"], kTfLiteSessionParams);
  config->joint_file = filename_of(refs["joint"], kTfLiteSessionParams);
  config->joint_prior_file =
      filename_of(refs["joint_prior"], kTfLiteSessionParams);
  config->symbols_file = filename_of(refs["symbols"], kSymbolTableParams);
  fe.mean_stddev_file = filename_of(mean_stddev_ref, kProtoParams);
  config->vad.model_file = filename_of(vad_model_ref, kNeuralNetworkParams);
  config->vad.mean_stddev_file = filename_of(vad_mean_stddev_ref, kProtoParams);
  auto is_capitalization = [](const std::string& classname) {
    return classname == "GeneralCapitalizationNormalizer" ||
           classname == "RuntimeRoadrunnerCapitalizationNormalizer";
  };
  for (const auto& [name, def] : resources) {
    if (!is_capitalization(def.classname)) continue;
    // Both classes keep their fields in one params message: 1 and 2 for the
    // general normalizer, 12 and up for the Roadrunner model.
    ProtoReader c(FindExtension(def.params, kCapitalizationParams));
    while (c.Next()) {
      switch (c.field()) {
        case 1: config->capitalization_transducer_file = c.str(); break;
        case 2: config->capitalization_lm_file = c.str(); break;
        case 12: config->capitalization_model.file = c.str(); break;
        case 14: config->capitalization_model.window = static_cast<int>(c.int64()); break;
        case 15: config->capitalization_model.left = static_cast<int>(c.int64()); break;
        case 16: config->capitalization_model.right = static_cast<int>(c.int64()); break;
      }
    }
  }
  for (const auto& [name, def] : resources) {
    // DenormalizerResourceList {1: {1: resource_name, 2: denormalizer_type}}.
    ProtoReader list(FindExtension(def.params, kDenormalizerResourceList));
    bool capitalization_seen = false;
    for (ProtoReader scan(FindExtension(def.params, kDenormalizerResourceList)); scan.Next();) {
      if (scan.field() != 1) continue;
      auto it = resources.find(FirstString(scan.bytes(), 1));
      capitalization_seen |= it != resources.end() && is_capitalization(it->second.classname);
    }
    // Whether the capitalization normalizer is still ahead in the list.
    bool capitalization_ahead = capitalization_seen;
    while (list.Next()) {
      if (list.field() != 1) continue;
      ProtoReader ref(list.bytes());
      std::string resource;
      int type = 0;
      while (ref.Next()) {
        if (ref.field() == 1) resource = ref.str();
        if (ref.field() == 2) type = static_cast<int>(ref.varint());
      }
      auto it = resources.find(resource);
      if (it == resources.end()) continue;
      if (is_capitalization(it->second.classname)) capitalization_ahead = false;
      const std::string file = FirstString(FindExtension(it->second.params, kFstNormalizerParams), 1);
      if (it->second.classname == "PunctuationNormalizer") {
        // {2: converter_config_file, 3: neural_network_ref, 4: symbol_file}.
        std::string_view p = FindExtension(it->second.params, kPunctuationNormalizerParams);
        config->spoken_punctuation.type = type;
        config->spoken_punctuation.converter_file = FirstString(p, 2);
        config->spoken_punctuation.symbol_file = FirstString(p, 4);
        config->spoken_punctuation.model_file = filename_of(FirstString(p, 3), kNeuralNetworkParams);
      } else if (it->second.classname == "PortablePqrnnSpokenPunctuationNormalizer") {
        // {1: model_file, 2: inference_settings}.
        std::string_view p = FindExtension(it->second.params, kPqrnnSpokenPunctuationParams);
        PipelineConfig::Denormalizer d{FirstString(p, 1), type, capitalization_ahead};
        d.spoken_punctuation_settings = FirstString(p, 2);
        if (!d.file.empty()) config->denormalizers.push_back(d);
      } else if (type == kPornDenormalizer) {
        config->offensive_word_masker_file = file;
      } else if (!file.empty() && (it->second.classname == "FstNormalizer" ||
                                   it->second.classname == "ClassNormalizer")) {
        bool listed = false;
        for (const auto& d : config->denormalizers) listed |= d.file == file;
        if (!listed) {
          config->denormalizers.push_back(
              {file, type, capitalization_ahead, it->second.classname == "ClassNormalizer"});
        }
      }
    }
  }
  auto search_params = [&](const std::string& ref, DecoderConfig* d) {
    auto it = resources.find(ref);
    if (it == resources.end()) return;
    // FstSearchParams is the only extension in that Params message.
    ProtoReader p(it->second.params);
    while (p.Next()) {
      if (p.wire_type() == kLengthDelimited) ParseSearchParams(p.bytes(), d);
    }
  };
  search_params(refs["search"], &dec);
  if (!cascade_encoder_ref.empty() && decoders_seen > 1) {
    CascadeConfig& c = config->cascade;
    c.encoder.file = filename_of(cascade_encoder_ref, kTfLiteSessionParams);
    domain_ids(cascade_domains_ref, &c.encoder);
    c.decoder_file = filename_of(cascade_refs["decoder"], kTfLiteSessionParams);
    c.joint_file = filename_of(cascade_refs["joint"], kTfLiteSessionParams);
    c.joint_prior_file = filename_of(cascade_refs["joint_prior"], kTfLiteSessionParams);
    search_params(cascade_refs["search"], &c.decoder);
    if (c.decoder_file.empty() || c.joint_file.empty()) c.encoder.file.clear();
  }

  for (const auto& [name, def] : resources) {
    if (FindTwiddlerParams(def.params, resources, &config->biasing)) break;
  }

  if (config->encoder.file.empty() || config->decoder_file.empty() ||
      config->joint_file.empty() || config->symbols_file.empty()) {
    *error = "config " + path + " lacks an end-to-end transducer model";
    return false;
  }
  return true;
}

}  // namespace soda

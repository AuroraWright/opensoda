/*
 * C API compatible with Chrome's libsoda (SODA 1.2.x).
 *
 * Exports the same entry points Chrome resolves from libsoda.so. The
 * configuration and responses are serialized protos from Chromium's
 * chrome/services/speech/soda/proto/soda_api.proto
 * (speech.soda.chrome.ExtendedSodaConfigMsg and SodaResponse).
 *
 * Callbacks run on the recognizer's worker thread.
 */

#ifndef OPENSODA_SODA_API_H_
#define OPENSODA_SODA_API_H_

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define SODA_EXPORT __declspec(dllexport)
#else
#define SODA_EXPORT __attribute__((visibility("default")))
#endif

/* Receives a serialized SodaResponse, its length, and callback_handle. */
typedef void (*SerializedSodaEventHandler)(const char* response, int length,
                                           void* callback_handle);

typedef struct {
  /* A serialized ExtendedSodaConfigMsg. Not owned. */
  const char* soda_config;
  int soda_config_size;
  SerializedSodaEventHandler callback;
  void* callback_handle;
} SerializedSodaConfig;

typedef struct {
  /* A serialized RecognitionContext. */
  const char* recognition_context;
  int recognition_context_size;
} RecognitionContext;

/* Returns NULL if the language pack cannot be loaded. */
SODA_EXPORT void* CreateExtendedSodaAsync(SerializedSodaConfig config);
SODA_EXPORT void DeleteExtendedSodaAsync(void* soda_async_handle);

/* Interleaved 16-bit little-endian PCM at the configured rate/channels. */
SODA_EXPORT void ExtendedAddAudio(void* soda_async_handle, const char* audio_buffer,
                                  int audio_buffer_size);

/* Starts (or restarts) recognition. */
SODA_EXPORT void ExtendedSodaStart(void* soda_async_handle);
/* Stops recognition, finalizing the current result. */
SODA_EXPORT void ExtendedSodaStop(void* soda_async_handle);
/* Marks the end of the audio stream; the final result follows. */
SODA_EXPORT void ExtendedSodaMarkDone(void* soda_async_handle);

/* Replaces the phrases recognition is biased toward (packs with a contextual
   biasing model only). */
SODA_EXPORT void UpdateRecognitionContext(void* soda_async_handle,
                                          RecognitionContext context);

#ifdef __cplusplus
}
#endif

#endif /* OPENSODA_SODA_API_H_ */

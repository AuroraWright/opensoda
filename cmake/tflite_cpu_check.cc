/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Modified by the opensoda authors: detects the dot product instructions on
// macOS as well.
// Replaces TFLite's kernels/internal/optimized/cpu_check.cc (Apache-2.0),
// which detects the Arm dot product instructions only on Linux. On Apple
// silicon its hybrid int8 kernels then skip their SDOT paths, which give the
// same results several times faster; libsoda's TFLite detects them on macOS
// too (hw.optional.arm.FEAT_DotProd).
#include "tensorflow/lite/kernels/internal/optimized/cpu_check.h"

#if defined __linux__ && defined __aarch64__
#include <sys/auxv.h>
#endif
#if defined __APPLE__ && defined __aarch64__
#include <sys/sysctl.h>
#endif

namespace tflite {

bool DetectArmNeonDotprod() {
#if defined __linux__ && defined __aarch64__
  // HWCAP_ASIMDDP.
  return getauxval(AT_HWCAP) & (1 << 20);
#elif defined __APPLE__ && defined __aarch64__
  int value = 0;
  size_t size = sizeof(value);
  return sysctlbyname("hw.optional.arm.FEAT_DotProd", &value, &size, nullptr, 0) == 0 &&
         value != 0;
#else
  return false;
#endif
}

}  // namespace tflite

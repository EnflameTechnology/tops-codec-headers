/**
 * Copyright 2023 The Enflame Tech Company. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef TOPS_INCLUDE_TOPS_TOPS_EXT_EXPERIMENTAL_H
#define TOPS_INCLUDE_TOPS_TOPS_EXT_EXPERIMENTAL_H
#include <tops/tops_runtime_api.h>
#if defined(__cplusplus)
#include <tuple>
#include <type_traits>
#endif

// Structure definitions:
#ifdef __cplusplus
extern "C" {
#endif

/**
 *     @defgroup GlobalDefsExt Global enum and defines
 *     @{
 *
 */

/**
 * topsResourceRequestV2 descriptor
 * @see topsSetDeviceAndResourceReservation
 */
typedef struct topsResourceRequestV2_st {
  uint64_t reserved_threads;
} topsResourceRequestV2_t;

// Doxygen end group GlobalDefsExt
/**  @} */

/**
 *  @addtogroup Extension TOPS API
 *  @{
 *  @ingroup Extension
 */

/**
 * @brief Set default device to be used for subsequent tops API calls from this thread.
 *
 * @param[in] deviceId Valid device in range 0...(topsGetDeviceCount()-1).
 * @param[in] resRequest Create a device by applying for a specified number of threads.
 *
 * Sets @p device as the default device for the calling host thread.  Valid device id's are 0...
 * (topsGetDeviceCount()-1).
 *
 * Many TOPS APIs implicitly use the "default device" :
 *
 * - Any device memory subsequently allocated from this host thread (using topsMalloc) will be
 * allocated on device.
 * - Any streams or events created from this host thread will be associated with device.
 * - Any kernels launched from this host thread (using topsLaunchKernel) will be executed on device
 * (unless a specific stream is specified, in which case the device associated with that stream will
 * be used).
 *
 * This function may be called from any host thread.  Multiple host threads may use the same device.
 * This function does no synchronization with the previous or new device, and has very little
 * runtime overhead. Applications can use topsSetDevice to quickly switch the default device before
 * making a TOPS runtime call which uses the default device.
 *
 * The default device is stored in thread-local-storage for each thread.
 * Thread-pool implementations may inherit the default device of the previous thread.  A good
 * practice is to always call topsSetDevice at the start of TOPS coding sequency to establish a known
 * standard device.
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice, #topsErrorDeviceAlreadyInUse
 *
 * @see topsGetDevice, topsGetDeviceCount
 */
TOPS_PUBLIC_API
topsError_t topsSetDeviceAndResourceReservation ( int  deviceId, topsResourceRequestV2_t *resRequest );
/**
 * @}
 */

#ifdef __cplusplus
} /* extern "c" */
#endif

#endif  // #iidef TOPS_INCLUDE_TOPS_TOPS_EXT_EXPERIMENTAL_H

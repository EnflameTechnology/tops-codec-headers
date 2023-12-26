/**
 * Copyright 2020 The Enflame Tech Company. All Rights Reserved.
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

/**
 *  @file  host_defines.h
 *  @brief TODO-doc
 */

#ifndef TOPS_INCLUDE_TOPS_HOST_DEFINES_H
#define TOPS_INCLUDE_TOPS_HOST_DEFINES_H

// Add guard to Generic Grid Launch method
#ifndef GENERIC_GRID_LAUNCH
#define GENERIC_GRID_LAUNCH 1
#endif

#if defined(__clang__) && defined(__TOPS__)

#if !__CLANG_TOPS_H_INCLUDED__

#define __constant__           __attribute__((constant, section(".rodata")))
#define __device__             __attribute__((device))
#define __global__             __attribute__((global))
#define __host__               __attribute__((host))
#define __sp__                 __attribute__((sp))
#define __shared__             __attribute__((shared))
#define __shared_dte__         __attribute__((shared_dte))
#define __private_dte__        __attribute__((private_dte))
#define __private__            __attribute__((address_space(5)))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __noinline__           __attribute__((noinline))
#define __forceinline__        inline __attribute__((always_inline))
#define __valigned__           __attribute__((aligned(128)))

#endif // !__CLANG_TOPS_H_INCLUDED__

#else

/**
 * Function and kernel markers
 */
#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

#ifndef __noinline__
#define __noinline__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

#ifndef __shared__
#define __shared__
#endif
#ifndef __constant__
#define __constant__
#endif

#endif

#endif

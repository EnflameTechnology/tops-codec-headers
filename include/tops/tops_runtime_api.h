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
 * @file tops_runtime_api.h
 *
 * @brief Defines the API signatures for TOPS runtime.
 * This file can be compiled with a standard compiler.
 */

#ifndef TOPS_INCLUDE_TOPS_TOPS_RUNTIME_API_H
#define TOPS_INCLUDE_TOPS_TOPS_RUNTIME_API_H
#include <string.h>  // for getDeviceProp
#include <limits.h>
#include <tops/tops_version.h>
#include <tops/tops_common.h>

// TODO: hack for all targets
#ifndef __TOPS_PLATFORM_ENFLAME__
#define __TOPS_PLATFORM_ENFLAME__
#endif

enum {
    TOPS_SUCCESS = 0,
    TOPS_ERROR_INVALID_VALUE,
    TOPS_ERROR_NOT_INITIALIZED,
    TOPS_ERROR_LAUNCH_OUT_OF_RESOURCES
};

/**
 * topsDeviceArch
 *
 */
typedef struct {
    unsigned feature;
} topsDeviceArch_t;

/**
 * topsDeviceProp
 *
 */
typedef struct topsDeviceProp_t {
    char name[256];            ///< Device name.
    size_t totalGlobalMem;     ///< Size of global memory region (in bytes).
    size_t sharedMemPerBlock;  ///< Size of shared memory region (in bytes).
    int maxThreadsPerBlock;    ///< Max work items per work group or workgroup max size.
    int maxThreadsDim[3];      ///< Max number of threads in each dimension (XYZ) of a block.
    int maxGridSize[3];        ///< Max grid dimensions (XYZ).
    int clockRate;             ///< Max clock frequency of the multiProcessors in khz.
    int memoryClockRate;       ///< Max global memory clock frequency in khz.
    int memoryBusWidth;        ///< Global memory bus width in bits.
    int major;                 ///< Major compute capability.
    int minor;                 ///< Minor compute capability.
    int multiProcessorCount;          ///< Number of multi-processors (compute units).
    int maxThreadsPerMultiProcessor;  ///< Maximum resident threads per multi-processor.
    int computeMode;                  ///< Compute mode.
    int clockInstructionRate;  ///< Frequency in khz of the timer used by the device-side "clock*"
                               ///< instructions.
    topsDeviceArch_t arch;      ///< Architectural feature flags.
    int concurrentKernels;     ///< Device can possibly execute multiple kernels concurrently.
    int pciDomainID;           ///< PCI Domain ID
    int pciBusID;              ///< PCI Bus ID.
    int pciDeviceID;           ///< PCI Device ID.
    size_t maxSharedMemoryPerMultiProcessor;  ///< Maximum Shared Memory Per Multiprocessor.
    int canMapHostMemory;                     ///< Check whether TOPS can map host memory
    char gcuArchName[256];                    ///< ENFLAME GCU Arch Name.
    int cooperativeLaunch;            ///< TOPS device supports cooperative launch
    int cooperativeMultiDeviceLaunch; ///< TOPS device supports cooperative launch on multiple devices
    int kernelExecTimeoutEnabled;    ///<Run time limit for kernels executed on the device
    int ECCEnabled;                  ///<Device has ECC support enabled
    int isLargeBar;                  ///< 1: if it is a large PCI bar device, else 0
    int managedMemory;               ///< Device supports allocating managed memory on this system
    int directManagedMemAccessFromHost; ///< Host can directly access managed memory on the device without migration
    int concurrentManagedAccess;     ///< Device can coherently access managed memory concurrently with the CPU
    int pageableMemoryAccess;        ///< Device supports coherently accessing pageable memory
                                     ///< without calling topsHostRegister on it
    // NOTE: tops extern
    int mc_channel;            ///< Memory Control Channel count.
    size_t localMemPerThread;  ///< Size of local memory region (in bytes).
    char localUuidId[16];      ///< uuid of local device
    char remoteUuidId[6][16];  ///< uuid of remote devices
} topsDeviceProp_t;

/**
 * Memory type (for pointer attributes)
 */
typedef enum topsMemoryType {
    topsMemoryTypeHost = 0x1,    ///< Memory is physically located on host
    topsMemoryTypeDevice = 0x2,  ///< Memory is physically located on device. (see deviceId for specific
                          ///< device)
    // NOTE: tops extern
    topsMemoryTypeScatter = 0x4,  ///< Memory is scatter memory type
    topsMemoryTypeLazy = 0x8,  ///< Memory is scatter memory type
                                    ///< which is lazy created

    topsMemoryTypeUnified  ///< Not used currently
} topsMemoryType;

/**
 * Pointer attributes
 */
typedef struct topsPointerAttribute_t {
    enum topsMemoryType memoryType;
    int device;
    void* device_pointer;
    void* host_pointer;
    int isManaged;
    unsigned allocationFlags;
} topsPointerAttribute_t;

/**
 * Topology Link Type
 */
typedef enum topsTopologyMapType {
    topsTopologyMapTypeAuto     = 0x0,  ///< Devices are linked on the priority
    topsTopologyMapTypeLocal    = 0x1,  ///< Devices are linked through local memory
    topsTopologyMapTypeEsl      = 0x2,  ///< Devices are linked through ESL
    topsTopologyMapTypeP2P      = 0x3,  ///< Devices are linked through P2P
    topsTopologyMapTypeClsAsDev = 0x4,  ///< Devices are linked through Cluster as Device
} topsTopologyMapType;

// hack to get these to show up in Doxygen:
/**
 *     @defgroup GlobalDefs Global enum and defines
 *     @{
 *
 */

// Ignoring error-code return values from tops APIs is discouraged. On C++17,
// we can make that yield a warning
#if __cplusplus >= 201703L
#define __TOPS_NODISCARD [[nodiscard]]
#else
#define __TOPS_NODISCARD
#endif

/*
 * @brief topsError_t
 * @enum
 * @ingroup Enumerations
 */

typedef enum __TOPS_NODISCARD topsError_t {
    topsSuccess = 0,  ///< Successful completion.
    topsErrorInvalidValue = 1,  ///< One or more of the parameters passed to the API call is NULL
                               ///< or not in an acceptable range.
    topsErrorOutOfMemory = 2,
    // Deprecated
    topsErrorMemoryAllocation = 2,  ///< Memory allocation error.
    topsErrorNotInitialized = 3,
    // Deprecated
    topsErrorInitializationError = 3,
    topsErrorDeinitialized = 4,
    topsErrorProfilerDisabled = 5,
    topsErrorProfilerNotInitialized = 6,
    topsErrorProfilerAlreadyStarted = 7,
    topsErrorProfilerAlreadyStopped = 8,
    topsErrorInvalidConfiguration = 9,
    topsErrorInvalidPitchValue = 12,
    topsErrorInvalidSymbol = 13,
    topsErrorInvalidDevicePointer = 17,  ///< Invalid Device Pointer
    topsErrorInvalidMemcpyDirection = 21,  ///< Invalid memory copy direction
    topsErrorInsufficientDriver = 35,
    topsErrorMissingConfiguration = 52,
    topsErrorPriorLaunchFailure = 53,
    topsErrorInvalidDeviceFunction = 98,
    topsErrorNoDevice = 100,  ///< Call to topsGetDeviceCount returned 0 devices
    topsErrorInvalidDevice = 101,  ///< DeviceID must be in range 0...#compute-devices.
    topsErrorInvalidImage = 200,
    topsErrorInvalidContext = 201,  ///< Produced when input context is invalid.
    topsErrorContextAlreadyCurrent = 202,
    topsErrorMapFailed = 205,
    // Deprecated
    topsErrorMapBufferObjectFailed = 205,  ///< Produced when the IPC memory attach failed from ROCr.
    topsErrorUnmapFailed = 206,
    topsErrorArrayIsMapped = 207,
    topsErrorAlreadyMapped = 208,
    topsErrorNoBinaryForGcu = 209,
    topsErrorAlreadyAcquired = 210,
    topsErrorNotMapped = 211,
    topsErrorNotMappedAsArray = 212,
    topsErrorNotMappedAsPointer = 213,
    topsErrorECCNotCorrectable = 214,
    topsErrorUnsupportedLimit = 215,
    topsErrorContextAlreadyInUse = 216,
    topsErrorPeerAccessUnsupported = 217,
    topsErrorInvalidKernelFile = 218,
    topsErrorInvalidGraphicsContext = 219,
    topsErrorInvalidSource = 300,
    topsErrorFileNotFound = 301,
    topsErrorSharedObjectSymbolNotFound = 302,
    topsErrorSharedObjectInitFailed = 303,
    topsErrorOperatingSystem = 304,
    topsErrorInvalidHandle = 400,
    // Deprecated
    topsErrorInvalidResourceHandle = 400,  ///< Resource handle (topsEvent_t or topsStream_t) invalid.
    topsErrorIllegalState = 401, ///< Resource required is not in a valid state to perform operation.
    topsErrorNotFound = 500,
    topsErrorNotReady = 600,  ///< Indicates that asynchronous operations enqueued earlier are not
                             ///< ready.  This is not actually an error, but is used to distinguish
                             ///< from topsSuccess (which indicates completion).  APIs that return
                             ///< this error include topsEventQuery and topsStreamQuery.
    topsErrorIllegalAddress = 700,
    topsErrorLaunchOutOfResources = 701,  ///< Out of resources error.
    topsErrorLaunchTimeOut = 702,
    topsErrorPeerAccessAlreadyEnabled =
        704,  ///< Peer access was already enabled from the current device.
    topsErrorPeerAccessNotEnabled =
        705,  ///< Peer access was never enabled from the current device.
    topsErrorSetOnActiveProcess = 708,
    topsErrorContextIsDestroyed = 709,
    topsErrorAssert = 710,  ///< Produced when the kernel calls assert.
    topsErrorHostMemoryAlreadyRegistered =
        712,  ///< Produced when trying to lock a page-locked memory.
    topsErrorHostMemoryNotRegistered =
        713,  ///< Produced when trying to unlock a non-page-locked memory.
    topsErrorLaunchFailure =
        719,  ///< An exception occurred on the device while executing a kernel.
    topsErrorCooperativeLaunchTooLarge =
        720,  ///< This error indicates that the number of blocks launched per grid for a kernel
              ///< that was launched via cooperative launch APIs exceeds the maximum number of
              ///< allowed blocks for the current device
    topsErrorNotSupported = 801,  ///< Produced when the tops API is not supported/implemented
    topsErrorStreamCaptureUnsupported = 900,  ///< The operation is not permitted when the stream
                                             ///< is capturing.
    topsErrorStreamCaptureInvalidated = 901,  ///< The current capture sequence on the stream
                                             ///< has been invalidated due to a previous error.
    topsErrorStreamCaptureMerge = 902,  ///< The operation would have resulted in a merge of
                                       ///< two independent capture sequences.
    topsErrorStreamCaptureUnmatched = 903,  ///< The capture was not initiated in this stream.
    topsErrorStreamCaptureUnjoined = 904,  ///< The capture sequence contains a fork that was not
                                          ///< joined to the primary stream.
    topsErrorStreamCaptureIsolation = 905,  ///< A dependency would have been created which crosses
                                           ///< the capture sequence boundary. Only implicit
                                           ///< in-stream ordering dependencies
                                           ///< are allowed
                                           ///< to cross the boundary
    topsErrorStreamCaptureImplicit = 906,  ///< The operation would have resulted in a disallowed
                                          ///< implicit dependency on a current capture sequence
                                          ///< from topsStreamLegacy.
    topsErrorCapturedEvent = 907,  ///< The operation is not permitted on an event which was last
                                  ///< recorded in a capturing stream.
    topsErrorStreamCaptureWrongThread = 908,  ///< A stream capture sequence not initiated with
                                             ///< the topsStreamCaptureModeRelaxed argument to
                                             ///< topsStreamBeginCapture was passed to
                                             ///< topsStreamEndCapture in a different thread.
    topsErrorGraphExecUpdateFailure = 910,  ///< This error indicates that the graph update
                                           ///< not performed because it included changes which
                                           ///< violated constraints specific to instantiated graph
                                           ///< update.
    topsErrorUnknown = 999,  //< Unknown error.
    // HSA Runtime Error Codes start here.
    topsErrorRuntimeMemory = 1052,  ///< HSA runtime memory call returned error.  Typically not seen
                                   ///< in production systems.
    topsErrorRuntimeOther = 1053,  ///< HSA runtime call other than memory returned error.  Typically
                                  ///< not seen in production systems.
    topsErrorTbd  ///< Marker that more error codes are needed.
} topsError_t;

#undef __TOPS_NODISCARD

/*
 * @brief topsDeviceAttribute_t
 * @enum
 * @ingroup Enumerations
 */
typedef enum topsDeviceAttribute_t {
    topsDeviceAttributeClockRate,                        ///< Peak clock frequency in kilohertz.
    topsDeviceAttributeComputeCapabilityMajor,           ///< Major compute capability version number.
    topsDeviceAttributeComputeCapabilityMinor,           ///< Minor compute capability version number.
    topsDeviceAttributeMaxBlockDimX,                     ///< Max block size in width.
    topsDeviceAttributeMaxBlockDimY,                     ///< Max block size in height.
    topsDeviceAttributeMaxBlockDimZ,                     ///< Max block size in depth.
    topsDeviceAttributeMaxGridDimX,                      ///< Max grid size  in width.
    topsDeviceAttributeMaxGridDimY,                      ///< Max grid size  in height.
    topsDeviceAttributeMaxGridDimZ,                      ///< Max grid size  in depth.
    topsDeviceAttributeMaxThreadsPerBlock,               ///< Maximum number of threads per block.
    topsDeviceAttributeMultiprocessorCount,              ///< Number of multiprocessors on the device.
    topsDeviceAttributeMaxSharedMemoryPerBlock,          ///< Maximum shared memory available per block in bytes.

    //NOTE: tops extern
    topsDeviceAttributeEnflameSpecificBegin = 10000,
    topsDeviceAttributeEnflameMcChannel,                 ///< Memory Control Channel count.
} topsDeviceAttribute_t;

enum topsComputeMode {
    topsComputeModeDefault = 0,
    topsComputeModeExclusive = 1,
    topsComputeModeProhibited = 2,
    topsComputeModeExclusiveProcess = 3
};

struct itopsStream_t {};
struct itopsEvent_t {};
struct itopsExecutable_t {};
struct itopsResource_t {};
struct itopsCtx_t {};
struct itopsTensor_t {};

/**
 * @}
 */

#if (defined(__TOPS_PLATFORM_TOPSCC__) || defined(__TOPS_PLATFORM_ENFLAME__))

#include <stdint.h>
#include <stddef.h>
#ifndef GENERIC_GRID_LAUNCH
#define GENERIC_GRID_LAUNCH 1
#endif
#include <tops/host_defines.h>
#include <tops/driver_types.h>
#if defined(_MSC_VER)
#define DEPRECATED(msg) __declspec(deprecated(msg))
#else // !defined(_MSC_VER)
#define DEPRECATED(msg) __attribute__ ((deprecated(msg)))
#endif // !defined(_MSC_VER)
#define DEPRECATED_MSG "This API is marked as deprecated and may not be supported in future releases."
#define TOPS_LAUNCH_PARAM_BUFFER_POINTER ((void*)0x01)
#define TOPS_LAUNCH_PARAM_BUFFER_SIZE ((void*)0x02)
#define TOPS_LAUNCH_PARAM_END ((void*)0x03)
#ifdef __cplusplus
  #define __dparm(x) \
          = x
#else
  #define __dparm(x)
#endif
#ifdef __GNUC__
#pragma GCC visibility push (default)
#endif

// Structure definitions:
#ifdef __cplusplus
extern "C" {
#endif
//---
// API-visible structures
typedef struct itopsCtx_t* topsCtx_t;
// Note many APIs also use integer deviceIds as an alternative to the device pointer:
typedef int topsDevice_t;
typedef enum topsDeviceP2PAttr {
  topsDevP2PAttrPerformanceRank = 0,
  topsDevP2PAttrAccessSupported,
  topsDevP2PAttrNativeAtomicSupported,
  topsDevP2PAttrTopsArrayAccessSupported
} topsDeviceP2PAttr;
typedef struct itopsStream_t* topsStream_t;
#define topsIpcMemLazyEnablePeerAccess 0
#define TOPS_IPC_HANDLE_SIZE 64
typedef struct topsIpcMemHandle_st {
    char reserved[TOPS_IPC_HANDLE_SIZE];
} topsIpcMemHandle_t;
typedef struct topsIpcEventHandle_st {
    char reserved[TOPS_IPC_HANDLE_SIZE];
} topsIpcEventHandle_t;
typedef struct itopsModule_t* topsModule_t;
typedef struct itopsModuleSymbol_t* topsFunction_t;
typedef struct itopMemPoolHandle_t* topsMemPool_t;

typedef struct topsFuncAttributes {
    int binaryVersion;
    int cacheModeCA;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxDynamicSharedSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int preferredShmemCarveout;
    int ptxVersion;
    size_t sharedSizeBytes;
} topsFuncAttributes;
typedef struct itopsEvent_t* topsEvent_t;
enum topsLimit_t {
    topsLimitPrintfFifoSize = 0x01,
    topsLimitMallocHeapSize = 0x02,
};
/**
 * @addtogroup GlobalDefs More
 * @{
 */
//Flags that can be used with topsStreamCreateWithFlags.
/** Default stream creation flags. These are used with topsStreamCreate().*/
#define topsStreamDefault  0x00

/** Stream does not implicitly synchronize with null stream.*/
#define topsStreamNonBlocking 0x01

/* Stream blocks until callback is completed. */
#define topsStreamCallbackBlocking 0x2

/* Stream can not be merged with other streams. */
#define topsStreamNonMerging 0x4

//Flags that can be used with topsEventCreateWithFlags.
/** Default flags.*/
#define topsEventDefault 0x0

/** Waiting will yield CPU. Power-friendly and usage-friendly but may increase latency.*/
#define topsEventBlockingSync 0x1

/** Disable event's capability to record timing information. May improve performance.*/
#define topsEventDisableTiming  0x2

/** Event can support IPC. Warning: It is not supported in TOPS.*/
#define topsEventInterprocess 0x4

/** Event can only support record once. NOTE: tops extern*/
#define topsEventRecordOnce 0x8

/** Event can only support strong order. NOTE: tops extern */
#define topsEventStrongOrder 0x10

/** Disable event query/sync operation. */
#define topsEventDisableQueryAndSync 0x20

/** Use a device-scope release when recording this event. This flag is useful to obtain more
 * precise timings of commands between events. */
#define topsEventReleaseToDevice  0x40000000

/** Use a system-scope release when recording this event. This flag is useful to make
 * non-coherent host memory visible to the host. */
#define topsEventReleaseToSystem  0x80000000

//Flags that can be used with topsHostMalloc.
/** Default pinned memory allocation on the host.*/
#define topsHostMallocDefault 0x0

/** Memory is considered allocated by all contexts.*/
#define topsHostMallocPortable 0x1

/** Map the allocation into the address space for the current device. The device pointer
 * can be obtained with #topsHostGetDevicePointer.*/
#define topsHostMallocMapped  0x2

/** Allocates the memory as write-combined. On some system configurations, write-combined allocation
 * may be transferred faster across the PCI Express bus, however, could have low read efficiency by
 * most CPUs. It's a good option for data transfer from host to device via mapped pinned memory.*/
#define topsHostMallocWriteCombined 0x4

/** Host memory allocation will follow numa policy set by user.*/
#define topsHostMallocNumaUser  0x20000000

/** Allocate coherent memory. Overrides TOPS_COHERENT_HOST_ALLOC for specific allocation.*/
#define topsHostMallocCoherent  0x40000000

/** Allocate non-coherent memory. Overrides TOPS_COHERENT_HOST_ALLOC for specific allocation.*/
#define topsHostMallocNonCoherent  0x80000000

/** Memory can be accessed by any stream on any device*/
#define topsMemAttachGlobal  0x01

/** Memory cannot be accessed by any stream on any device.*/
#define topsMemAttachHost    0x02

/** Memory can only be accessed by a single stream on the associated device.*/
#define topsMemAttachSingle  0x04

#define topsDeviceMallocDefault 0x0

/** Memory is allocated in fine grained region of device.*/
#define topsDeviceMallocFinegrained 0x1

/** Memory represents a HSA signal.*/
#define topsMallocSignalMemory 0x2

/** Memory allocation from top to down.*/
#define topsMallocTopDown 0x4

/** Memory allocation with forbidden to do fragment.*/
#define topsMallocForbidMergeMove 0x8

/** Memory allocation prefer to be allocated on high speed mem, if it can not be satisfied,
 *  then allocate on global device memory.
 */
#define topsMallocPreferHighSpeedMem 0x10

/** Memory allocation host accessable.*/
#define topsMallocHostAccessable 0x20


//Flags that can be used with topsHostRegister.
/** Memory is Mapped and Portable.*/
#define topsHostRegisterDefault 0x0

/** Memory is considered registered by all contexts.*/
#define topsHostRegisterPortable 0x1

/** Map the allocation into the address space for the current device. The device pointer
 * can be obtained with #topsHostGetDevicePointer.*/
#define topsHostRegisterMapped  0x2

/** Not supported.*/
#define topsHostRegisterIoMemory 0x4

/** Coarse Grained host memory lock.*/
#define topsExtHostRegisterCoarseGrained 0x8

/** Automatically select between Spin and Yield.*/
#define topsDeviceScheduleAuto 0x0

/** Dedicate a CPU core to spin-wait. Provides lowest latency, but burns a CPU core and may
 * consume more power.*/
#define topsDeviceScheduleSpin  0x1

/** Yield the CPU to the operating system when waiting. May increase latency, but lowers power
 * and is friendlier to other threads in the system.*/
#define topsDeviceScheduleYield  0x2
#define topsDeviceScheduleBlockingSync 0x4
#define topsDeviceScheduleMask 0x7
#define topsDeviceMapHost 0x8
#define topsDeviceLmemResizeToMax 0x16
/** Default TOPS array allocation flag.*/
#define topsArrayDefault 0x00
#define topsArrayLayered 0x01
#define topsArraySurfaceLoadStore 0x02
#define topsArrayCubemap 0x04
#define topsArrayTextureGather 0x08
#define topsOccupancyDefault 0x00
#define topsCooperativeLaunchMultiDeviceNoPreSync 0x01
#define topsCooperativeLaunchMultiDeviceNoPostSync 0x02
#define topsCpuDeviceId ((int)-1)
#define topsInvalidDeviceId ((int)-2)
//Flags that can be used with topsExtLaunch Set of APIs.
/** AnyOrderLaunch of kernels.*/
#define topsExtAnyOrderLaunch 0x01
// Flags to be used with topsStreamWaitValue32 and topsStreamWaitValue64.
#define topsStreamWaitValueGte 0x0
#define topsStreamWaitValueEq 0x1
#define topsStreamWaitValueAnd 0x2
#define topsStreamWaitValueNor 0x3
// Stream per thread
/** Implicit stream per application thread.*/
#define topsStreamPerThread ((topsStream_t)2)
/*
 * @brief TOPS Memory Advise values
 * @enum
 * @ingroup Enumerations
 */
typedef enum topsMemoryAdvise {
    topsMemAdviseSetReadMostly = 1,          ///< Data will mostly be read and only occasionally
                                            ///< be written to
    topsMemAdviseUnsetReadMostly = 2,        ///< Undo the effect of topsMemAdviseSetReadMostly
    topsMemAdviseSetPreferredLocation = 3,   ///< Set the preferred location for the data as
                                            ///< the specified device
    topsMemAdviseUnsetPreferredLocation = 4, ///< Clear the preferred location for the data
    topsMemAdviseSetAccessedBy = 5,          ///< Data will be accessed by the specified device,
                                            ///< so prevent page faults as much as possible
    topsMemAdviseUnsetAccessedBy = 6,        ///< Let TOPS to decide on the page faulting policy
                                            ///< for the specified device
    topsMemAdviseSetCoarseGrain = 100,       ///< The default memory model is fine-grain. That allows
                                            ///< coherent operations between host and device, while
                                            ///< executing kernels. The coarse-grain can be used
                                            ///< for data that only needs to be coherent at dispatch
                                            ///< boundaries for better performance
    topsMemAdviseUnsetCoarseGrain = 101      ///< Restores cache coherency policy back to fine-grain
} topsMemoryAdvise;
/*
 * @brief TOPS Coherency Mode
 * @enum
 * @ingroup Enumerations
 */
typedef enum topsMemRangeCoherencyMode {
    topsMemRangeCoherencyModeFineGrain = 0,      ///< Updates to memory with this attribute can be
                                                ///< done coherently from all devices
    topsMemRangeCoherencyModeCoarseGrain = 1,    ///< Writes to memory with this attribute can be
                                                ///< performed by a single device at a time
    topsMemRangeCoherencyModeIndeterminate = 2   ///< Memory region queried contains subregions with
                                                ///< both topsMemRangeCoherencyModeFineGrain and
                                                ///< topsMemRangeCoherencyModeCoarseGrain attributes
} topsMemRangeCoherencyMode;
/*
 * @brief TOPS range attributes
 * @enum
 * @ingroup Enumerations
 */
typedef enum topsMemRangeAttribute {
    topsMemRangeAttributeReadMostly = 1,         ///< Whether the range will mostly be read and
                                                ///< only occasionally be written to
    topsMemRangeAttributePreferredLocation = 2,  ///< The preferred location of the range
    topsMemRangeAttributeAccessedBy = 3,         ///< Memory range has topsMemAdviseSetAccessedBy
                                                ///< set for the specified device
    topsMemRangeAttributeLastPrefetchLocation = 4,///< The last location to where the range was
                                                ///< prefetched
    topsMemRangeAttributeCoherencyMode = 100,    ///< Returns coherency mode
                                                ///< @ref topsMemRangeCoherencyMode for the range
} topsMemRangeAttribute;
/**
 * @brief TOPS memory pool attributes
 * @enum
 * @ingroup Enumerations
 */
typedef enum topsMemPoolAttr
{
    /**
     * (value type = int)
     * Allow @p topsMemAllocAsync to use memory asynchronously freed
     * in another streams as long as a stream ordering dependency
     * of the allocating stream on the free action exists.
     * tops events and null stream interactions can create the required
     * stream ordered dependencies. (default enabled)
     */
    topsMemPoolReuseFollowEventDependencies   = 0x1,
    /**
     * (value type = int)
     * Allow reuse of already completed frees when there is no dependency
     * between the free and allocation. (default enabled)
     */
    topsMemPoolReuseAllowOpportunistic        = 0x2,
    /**
     * (value type = int)
     * Allow @p topsMemAllocAsync to insert new stream dependencies
     * in order to establish the stream ordering required to reuse
     * a piece of memory released by cuFreeAsync (default enabled).
     */
    topsMemPoolReuseAllowInternalDependencies = 0x3,
    /**
     * (value type = uint64_t)
     * Amount of reserved memory in bytes to hold onto before trying
     * to release memory back to the OS. When more than the release
     * threshold bytes of memory are held by the memory pool, the
     * allocator will try to release memory back to the OS on the
     * next call to stream, event or context synchronize. (default 0)
     */
    topsMemPoolAttrReleaseThreshold           = 0x4,
    /**
     * (value type = uint64_t)
     * Amount of backing memory currently allocated for the mempool.
     */
    topsMemPoolAttrReservedMemCurrent         = 0x5,
    /**
     * (value type = uint64_t)
     * High watermark of backing memory allocated for the mempool since the
     * last time it was reset. High watermark can only be reset to zero.
     */
    topsMemPoolAttrReservedMemHigh            = 0x6,
    /**
     * (value type = uint64_t)
     * Amount of memory from the pool that is currently in use by the application.
     */
    topsMemPoolAttrUsedMemCurrent             = 0x7,
    /**
     * (value type = uint64_t)
     * High watermark of the amount of memory from the pool that was in use by the application since
     * the last time it was reset. High watermark can only be reset to zero.
     */
    topsMemPoolAttrUsedMemHigh                = 0x8
} topsMemPoolAttr;
/**
 * @brief Specifies the type of location
 * @enum
 * @ingroup Enumerations
 */
 typedef enum topsMemLocationType {
    topsMemLocationTypeInvalid = 0,
    topsMemLocationTypeDevice = 1    ///< Device location, thus it's TOPS device ID
} topsMemLocationType;
/**
 * Specifies a memory location.
 *
 * To specify a gpu, set type = @p topsMemLocationTypeDevice and set id = the gpu's device ID
 */
typedef struct topsMemLocation {
    topsMemLocationType type;  ///< Specifies the location type, which describes the meaning of id
    int id;                   ///< Identifier for the provided location type @p topsMemLocationType
} topsMemLocation;
/**
 * @brief Specifies the memory protection flags for mapping
 * @enum
 * @ingroup Enumerations
 */
typedef enum topsMemAccessFlags {
    topsMemAccessFlagsProtNone      = 0,  ///< Default, make the address range not accessible
    topsMemAccessFlagsProtRead      = 1,  ///< Set the address range read accessible
    topsMemAccessFlagsProtReadWrite = 3   ///< Set the address range read-write accessible
} topsMemAccessFlags;
/**
 * Memory access descriptor
 */
typedef struct topsMemAccessDesc {
    topsMemLocation      location; ///< Location on which the accessibility has to change
    topsMemAccessFlags   flags;    ///< Accessibility flags to set
} topsMemAccessDesc;
/**
 * @brief Defines the allocation types
 * @enum
 * @ingroup Enumerations
 */
typedef enum topsMemAllocationType {
    topsMemAllocationTypeInvalid = 0x0,
    /** This allocation type is 'pinned', i.e. cannot migrate from its current
      * location while the application is actively using it
      */
    topsMemAllocationTypePinned  = 0x1,
    topsMemAllocationTypeMax     = 0x7FFFFFFF
} topsMemAllocationType;
/**
 * @brief Flags for specifying handle types for memory pool allocations
 * @enum
 * @ingroup Enumerations
 */
typedef enum topsMemAllocationHandleType {
    topsMemHandleTypeNone                    = 0x0,  ///< Does not allow any export mechanism
    topsMemHandleTypePosixFileDescriptor     = 0x1,  ///< Allows a file descriptor for exporting. Permitted only on POSIX systems
    topsMemHandleTypeWin32                   = 0x2,  ///< Allows a Win32 NT handle for exporting. (HANDLE)
    topsMemHandleTypeWin32Kmt                = 0x4   ///< Allows a Win32 KMT handle for exporting. (D3DKMT_HANDLE)
} topsMemAllocationHandleType;
/**
 * Specifies the properties of allocations made from the pool.
 */
typedef struct topsMemPoolProps {
    topsMemAllocationType       allocType;   ///< Allocation type. Currently must be specified as @p topsMemAllocationTypePinned
    topsMemAllocationHandleType handleTypes; ///< Handle types that will be supported by allocations from the pool
    topsMemLocation             location;    ///< Location where allocations should reside
    /**
     * Windows-specific LPSECURITYATTRIBUTES required when @p topsMemHandleTypeWin32 is specified
     */
    void*                       win32SecurityAttributes;
    unsigned char               reserved[64]; ///< Reserved for future use, must be 0
} topsMemPoolProps;
/**
 * Opaque data structure for exporting a pool allocation
 */
typedef struct topsMemPoolPtrExportData {
    unsigned char reserved[64];
} topsMemPoolPtrExportData;
/*
 * @brief topsJitOption
 * @enum
 * @ingroup Enumerations
 */
typedef enum topsJitOption {
    topsJitOptionMaxRegisters = 0,
    topsJitOptionThreadsPerBlock,
    topsJitOptionWallTime,
    topsJitOptionInfoLogBuffer,
    topsJitOptionInfoLogBufferSizeBytes,
    topsJitOptionErrorLogBuffer,
    topsJitOptionErrorLogBufferSizeBytes,
    topsJitOptionOptimizationLevel,
    topsJitOptionTargetFromContext,
    topsJitOptionTarget,
    topsJitOptionFallbackStrategy,
    topsJitOptionGenerateDebugInfo,
    topsJitOptionLogVerbose,
    topsJitOptionGenerateLineInfo,
    topsJitOptionCacheMode,
    topsJitOptionSm3xOpt,
    topsJitOptionFastCompile,
    topsJitOptionNumOptions
} topsJitOption;
/**
 * @warning these hints and controls are ignored.
 */
typedef enum topsFuncAttribute {
    topsFuncAttributeMaxDynamicSharedMemorySize = 8,
    topsFuncAttributePreferredSharedMemoryCarveout = 9,
    topsFuncAttributeMax
} topsFuncAttribute;
/**
 * @warning these hints and controls are ignored.
 */
typedef enum topsFuncCache_t {
    topsFuncCachePreferNone,    ///< no preference for shared memory or L1 (default)
    topsFuncCachePreferShared,  ///< prefer larger shared memory and smaller L1 cache
    topsFuncCachePreferL1,      ///< prefer larger L1 cache and smaller shared memory
    topsFuncCachePreferEqual,   ///< prefer equal size L1 cache and shared memory
} topsFuncCache_t;
/**
 * @warning these hints and controls are ignored.
 */
typedef enum topsSharedMemConfig {
    topsSharedMemBankSizeDefault,  ///< The compiler selects a device-specific value for the banking.
    topsSharedMemBankSizeFourByte,  ///< Shared mem is banked at 4-bytes intervals and performs best
                                   ///< when adjacent threads access data 4 bytes apart.
    topsSharedMemBankSizeEightByte  ///< Shared mem is banked at 8-byte intervals and performs best
                                   ///< when adjacent threads access data 4 bytes apart.
} topsSharedMemConfig;
/**
 * Struct for data in 3D
 *
 */
typedef union {
  struct {
    uint32_t x;
    uint32_t y;
    uint32_t z;
  };
  uint32_t data[3];
} uint3;
typedef struct dim3 {
    uint32_t x;  ///< x
    uint32_t y;  ///< y
    uint32_t z;  ///< z
#ifdef __cplusplus

#if defined(__clang__) && defined(__TOPS__)
#if !__CLANG_TOPS_H_INCLUDED__
    constexpr __host__ __device__ dim3(uint32_t _x = 1, uint32_t _y = 1,
                                       uint32_t _z = 1) : x(_x), y(_y), z(_z) {}
#else
    constexpr dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) :
                   x(_x), y(_y), z(_z) {}
#endif // !__CLANG_TOPS_H_INCLUDED__
#else
    constexpr dim3(uint32_t _x = 1, uint32_t _y = 1, uint32_t _z = 1) :
                   x(_x), y(_y), z(_z) {}
#endif

#endif
} dim3;
typedef struct topsLaunchParams_t {
    void* func;             ///< Device function symbol
    dim3 gridDim;           ///< Grid dimensions
    dim3 blockDim;          ///< Block dimensions
    void **args;            ///< Arguments
    size_t sharedMem;       ///< Shared memory
    topsStream_t stream;     ///< Stream identifier
} topsLaunchParams;
typedef enum topsExternalMemoryHandleType_enum {
  topsExternalMemoryHandleTypeOpaqueFd = 1,
  topsExternalMemoryHandleTypeOpaqueWin32 = 2,
  topsExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
  topsExternalMemoryHandleTypeD3D12Heap = 4,
  topsExternalMemoryHandleTypeD3D12Resource = 5,
  topsExternalMemoryHandleTypeD3D11Resource = 6,
  topsExternalMemoryHandleTypeD3D11ResourceKmt = 7,
} topsExternalMemoryHandleType;
typedef struct topsExternalMemoryHandleDesc_st {
  topsExternalMemoryHandleType type;
  union {
    int fd;
    struct {
      void *handle;
      const void *name;
    } win32;
  } handle;
  unsigned long long size;
  unsigned int flags;
} topsExternalMemoryHandleDesc;
typedef struct topsExternalMemoryBufferDesc_st {
  unsigned long long offset;
  unsigned long long size;
  unsigned int flags;
} topsExternalMemoryBufferDesc;
typedef void* topsExternalMemory_t;
typedef enum topsExternalSemaphoreHandleType_enum {
  topsExternalSemaphoreHandleTypeOpaqueFd = 1,
  topsExternalSemaphoreHandleTypeOpaqueWin32 = 2,
  topsExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
  topsExternalSemaphoreHandleTypeD3D12Fence = 4
} topsExternalSemaphoreHandleType;
typedef struct topsExternalSemaphoreHandleDesc_st {
  topsExternalSemaphoreHandleType type;
  union {
    int fd;
    struct {
      void* handle;
      const void* name;
    } win32;
  } handle;
  unsigned int flags;
} topsExternalSemaphoreHandleDesc;
typedef void* topsExternalSemaphore_t;
typedef struct topsExternalSemaphoreSignalParams_st {
  struct {
    struct {
      unsigned long long value;
    } fence;
    struct {
      unsigned long long key;
    } keyedMutex;
    unsigned int reserved[12];
  } params;
  unsigned int flags;
  unsigned int reserved[16];
} topsExternalSemaphoreSignalParams;
/**
 * External semaphore wait parameters, compatible with driver type
 */
typedef struct topsExternalSemaphoreWaitParams_st {
  struct {
    struct {
      unsigned long long value;
    } fence;
    struct {
      unsigned long long key;
      unsigned int timeoutMs;
    } keyedMutex;
    unsigned int reserved[10];
  } params;
  unsigned int flags;
  unsigned int reserved[16];
} topsExternalSemaphoreWaitParams;

/*
    * @brief TOPS Devices used by current OpenGL Context.
    * @enum
    * @ingroup Enumerations
    */
typedef enum topsGLDeviceList {
    topsGLDeviceListAll = 1,           ///< All tops devices used by current OpenGL context.
    topsGLDeviceListCurrentFrame = 2,  ///< Tops devices used by current OpenGL context in current
                                    ///< frame
    topsGLDeviceListNextFrame = 3      ///< Tops devices used by current OpenGL context in next
                                    ///< frame.
} topsGLDeviceList;

typedef struct _topsGraphicsResource topsGraphicsResource;

typedef topsGraphicsResource* topsGraphicsResource_t;

// Doxygen end group GlobalDefs
/**  @} */
//-------------------------------------------------------------------------------------------------
// The handle allows the async commands to use the stream even if the parent topsStream_t goes
// out-of-scope.
// typedef class itopsStream_t * topsStream_t;
/*
 * Opaque structure allows the true event (pointed at by the handle) to remain "live" even if the
 * surrounding topsEvent_t goes out-of-scope. This is handy for cases where the topsEvent_t goes
 * out-of-scope but the true event is being written by some async queue or device */
// typedef struct topsEvent_t {
//    struct itopsEvent_t *_handle;
//} topsEvent_t;
/**
 *  @defgroup API TOPS API
 *  @{
 *
 *  Defines the TOPS API.  See the individual sections for more information.
 */
/**
 *  @defgroup Driver Initialization and Version
 *  @{
 *  This section describes the initialization and version functions of TOPS runtime API.
 *
 */
/**
 * @brief Explicitly initializes the TOPS runtime.
 *
 * Most TOPS APIs implicitly initialize the TOPS runtime.
 * This API provides control over the timing of the initialization.
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuInit                   |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsInit(unsigned int flags);
/**
 * @brief Returns the approximate TOPS driver version.
 *
 * @param [out] driverVersion
 *
 * @returns #topsSuccess, #topsErrorInvalidValue
 *
 * The version is returned as (1000 major + 10 minor). For example, topsrider 2.2
 * would be represented by 2020.
 *
 * @see topsRuntimeGetVersion
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaDriverGetVersion     |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDriverGetVersion(int* driverVersion);
/**
 * @brief Returns the approximate TOPS Runtime version.
 *
 * @param [out] runtimeVersion
 *
 * @returns #topsSuccess, #topsErrorInvalidValue
 *
 * The version is returned as (1000 major + 10 minor). For example, topsrider 2.2
 * would be represented by 2020.
 *
 * @see topsDriverGetVersion
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaRuntimeGetVersion    |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsRuntimeGetVersion(int* runtimeVersion);
/**
 * @brief Returns a handle to a compute device
 * @param [out] device
 * @param [in] ordinal
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaGetDevice            |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDeviceGet(topsDevice_t* device, int ordinal);
/**
 * @brief Returns the compute capability of the device
 * @param [out] major
 * @param [out] minor
 * @param [in] device
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | No                       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDeviceComputeCapability(int* major, int* minor, topsDevice_t device);
/**
 * @brief Returns an identifier string for the device.
 * @param [out] name
 * @param [in] len
 * @param [in] device
 * @warning these versions are ignored.
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuDeviceGetName          |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDeviceGetName(char* name, int len, topsDevice_t device);
/**
 * @brief Returns a PCI Bus Id string for the device, overloaded to take int device ID.
 * @param [out] pciBusId
 * @param [in] len
 * @param [in] device
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaDeviceGetPCIBusId    |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDeviceGetPCIBusId(char* pciBusId, int len, int device);
/**
 * @brief Returns a handle to a compute device.
 * @param [out] device handle
 * @param [in] pciBusId
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaDeviceGetByPCIBusId  |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDeviceGetByPCIBusId(int* device, const char* pciBusId);
/**
 * @brief Returns the total amount of memory on the device.
 * @param [out] bytes
 * @param [in] device
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuDeviceTotalMem         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDeviceTotalMem(size_t* bytes, topsDevice_t device);
// doxygen end initialization
/**
 * @}
 */
/**
 *  @defgroup Device Device Management
 *  @{
 *  This section describes the device management functions of TOPS runtime API.
 */
/**
 * @brief Waits on all active streams on current device
 *
 * When this command is invoked, the host thread gets blocked until all the commands associated
 * with streams associated with the device. TOPS does not support multiple blocking modes (yet!).
 *
 * @returns #topsSuccess
 *
 * @see topsSetDevice, topsDeviceReset
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaDeviceSynchronize    |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDeviceSynchronize(void);
/**
 * @brief The state of current device is discarded and updated to a fresh state.
 *
 * Calling this function deletes all streams created, memory allocated, kernels running, events
 * created. Make sure that no other thread is using the device or streams, memory, kernels, events
 * associated with the current device.
 *
 * @returns #topsSuccess
 *
 * @see topsDeviceSynchronize
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaDeviceReset          |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDeviceReset(void);
/**
 * @brief Set default device to be used for subsequent tops API calls from this thread.
 *
 * @param[in] deviceId Valid device in range 0...(topsGetDeviceCount()-1).
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
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaSetDevice            |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsSetDevice(int deviceId);
/**
 * @brief Return the default device id for the calling host thread.
 *
 * @param [out] deviceId *deviceId is written with the default device
 *
 * TOPS maintains an default device for each thread using thread-local-storage.
 * This device is used implicitly for TOPS runtime APIs called by this thread.
 * topsGetDevice returns in * @p device the default device for the calling host thread.
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaGetDevice            |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsGetDevice(int* deviceId);
/**
 * @brief Return number of compute-capable devices.
 *
 * @param [output] count Returns number of compute-capable devices.
 *
 * @returns #topsSuccess, #topsErrorNoDevice
 *
 *
 * Returns in @p *count the number of devices that have ability to run compute commands.  If there
 * are no such devices, then @ref topsGetDeviceCount will return #topsErrorNoDevice. If 1 or more
 * devices can be found, then topsGetDeviceCount returns #topsSuccess.
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaGetDeviceCount       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsGetDeviceCount(int* count);
/**
 * @brief Query for a specific device attribute.
 *
 * @param [out] pi pointer to value to return
 * @param [in] attr attribute to query
 * @param [in] deviceId which device to query for information
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaDeviceGetAttribute   |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDeviceGetAttribute(int* pi, topsDeviceAttribute_t attr, int deviceId);
/**
 * @brief Returns device properties.
 *
 * @param [out] prop written with device properties
 * @param [in]  deviceId which device to query for information
 *
 * @return #topsSuccess, #topsErrorInvalidDevice
 * @bug tops always returns 0 for maxThreadsPerMultiProcessor
 * @bug tops always returns 0 for regsPerBlock
 * @bug tops always returns 0 for l2CacheSize
 *
 * Populates topsGetDeviceProperties with information for the specified device.
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaGetDeviceProperties  |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsGetDeviceProperties(topsDeviceProp_t* prop, int deviceId);
/**
 * @brief Get Resource limits of current device
 *
 * @param [out] pValue
 * @param [in]  limit
 *
 * @returns #topsSuccess, #topsErrorUnsupportedLimit, #topsErrorInvalidValue
 * Note: Currently, only topsLimitMallocHeapSize is available
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaDeviceGetLimit       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDeviceGetLimit(size_t* pValue, enum topsLimit_t limit);
/**
 * @brief Gets the flags set for current device
 *
 * @param [out] flags
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaGetDeviceFlags       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsGetDeviceFlags(unsigned int* flags);
/**
 * @brief The current device behavior is changed according the flags passed.
 *
 * @param [in] flags
 *
 * The schedule flags impact how TOPS waits for the completion of a command running on a device.
 *
 * topsDeviceScheduleSpin         : TOPS runtime will actively spin in the thread which submitted the
 * work until the command completes.  This offers the lowest latency, but will consume a CPU core
 * and may increase power.
 *
 * topsDeviceScheduleYield        : The TOPS runtime will yield the CPU to system so that other tasks
 * can use it.  This may increase latency to detect the completion but will consume less power and is
 * friendlier to other tasks in the system.
 *
 * topsDeviceScheduleBlockingSync : This is a synonym for topsDeviceScheduleYield.
 *
 * topsDeviceScheduleAuto         : Use a heuristic to select between Spin and Yield modes.  If the
 * number of TOPS contexts is greater than the number of logical processors in the system, use Spin
 * scheduling.  Else use Yield scheduling.
 *
 * topsDeviceMapHost              : Allow mapping host memory.  On GCU, this is always allowed and
 * the flag is ignored.
 *
 * topsDeviceLmemResizeToMax      : @warning GCU silently ignores this flag.
 *
 * @returns #topsSuccess, #topsErrorInvalidDevice, #topsErrorSetOnActiveProcess
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaSetDeviceFlags       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsSetDeviceFlags(unsigned flags);
/**
 * @brief Device which matches topsDeviceProp_t is returned
 *
 * @param [out] device The device ID
 * @param [in]  prop The device properties pointer
 *
 * @returns #topsSuccess, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaChooseDevice         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsChooseDevice(int* device, const topsDeviceProp_t* prop);

/**
 * @brief Gets an interprocess memory handle for an existing device memory
 *          allocation
 *
 * Takes a pointer to the base of an existing device memory allocation created
 * with topsMalloc and exports it for use in another process. This is a
 * lightweight operation and may be called multiple times on an allocation
 * without adverse effects.
 *
 * If a region of memory is freed with topsFree and a subsequent call
 * to topsMalloc returns memory with the same device address,
 * topsIpcGetMemHandle will return a unique handle for the
 * new memory.
 *
 * @param handle - Pointer to user allocated topsIpcMemHandle to return
 *                    the handle in.
 * @param devPtr - Base pointer to previously allocated device memory
 *
 * @returns
 * topsSuccess,
 * topsErrorInvalidHandle,
 * topsErrorOutOfMemory,
 * topsErrorMapFailed,
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaIpcGetMemHandle      |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsIpcGetMemHandle(topsIpcMemHandle_t* handle, void* devPtr);

/**
 * @brief Opens an interprocess memory handle exported from another process
 *          and returns a device pointer usable in the local process.
 *
 * Maps memory exported from another process with topsIpcGetMemHandle into
 * the current device address space. For contexts on different devices
 * topsIpcOpenMemHandle can attempt to enable peer access between the
 * devices as if the user called topsDeviceEnablePeerAccess. This behavior is
 * controlled by the topsIpcMemLazyEnablePeerAccess flag.
 * topsDeviceCanAccessPeer can determine if a mapping is possible.
 *
 * Contexts that may open topsIpcMemHandles are restricted in the following way.
 * topsIpcMemHandles from each device in a given process may only be opened
 * by one context per device per other process.
 *
 * Memory returned from topsIpcOpenMemHandle must be freed with
 * topsIpcCloseMemHandle.
 *
 * Calling topsFree on an exported memory region before calling
 * topsIpcCloseMemHandle in the importing context will result in undefined
 * behavior.
 *
 * @param devPtr - Returned device pointer
 * @param handle - topsIpcMemHandle to open
 * @param flags  - Flags for this operation. currently only flag 0 is supported
 *
 * @returns
 * topsSuccess,
 * topsErrorMapFailed,
 * topsErrorInvalidHandle,
 * topsErrorTooManyPeers
 *
 * @note No guarantees are made about the address returned in @p *devPtr.
 * In particular, multiple processes may not receive the same address for the
 * same @p handle.
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaIpcOpenMemHandle     |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsIpcOpenMemHandle(void** devPtr, topsIpcMemHandle_t handle,
                                 unsigned int flags);

/**
 * @brief Close memory mapped with topsIpcOpenMemHandle
 *
 * Unmaps memory returned by topsIpcOpenMemHandle. The original allocation
 * in the exporting process as well as imported mappings in other processes
 * will be unaffected.
 *
 * Any resources used to enable peer access will be freed if this is the
 * last mapping using them.
 *
 * @param devPtr - Device pointer returned by topsIpcOpenMemHandle
 *
 * @returns
 * topsSuccess,
 * topsErrorMapFailed,
 * topsErrorInvalidHandle,
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaIpcCloseMemHandle    |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsIpcCloseMemHandle(void* devPtr);

/**
 * @brief Gets an opaque interprocess handle for an event.
 *
 * This opaque handle may be copied into other processes and opened with
 * topsIpcOpenEventHandle. Then topsEventRecord, topsEventSynchronize,
 * topsEventQuery may be used in remote processes. The topsStreamWaitEvent is
 * called in local processes only. Operations on the imported event after
 * the exported event has been freed with topsEventDestroy will result in
 * undefined behavior.
 *
 * @param[out]  handle Pointer to topsIpcEventHandle to return the opaque event
 * handle
 * @param[in]   event  Event allocated with topsEventInterprocess and
 * topsEventDisableTiming flags
 *
 * @returns #topsSuccess, #topsErrorInvalidConfiguration, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaIpcGetEventHandle    |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsIpcGetEventHandle(topsIpcEventHandle_t* handle,
                                  topsEvent_t event);

/**
 * @brief Opens an interprocess event handle.
 *
 * Opens an interprocess event handle exported from another process with
 * topsIpcGetEventHandle. The returned topsEvent_t behaves like a locally
 * created event with the topsEventDisableTiming flag specified. This event need
 * be freed with topsEventDestroy. Operations on the imported event after the
 * exported event has been freed with topsEventDestroy will result in undefined
 * behavior. If the function is called within the same process where handle is
 * returned by topsIpcGetEventHandle, it will return topsErrorInvalidContext.
 *
 * @param[out]  event  Pointer to topsEvent_t to return the event
 * @param[in]   handle The opaque interprocess handle to open
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorInvalidContext
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaIpcOpenEventHandle   |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsIpcOpenEventHandle(topsEvent_t* event,
                                   topsIpcEventHandle_t handle);

/**
 * @brief Opens an interprocess event handle with topsTopologyMapType.
 *
 * Opens an interprocess event handle exported from another process with
 * topsIpcGetEventHandle. The returned topsEvent_t behaves like a locally
 * created event with the topsEventDisableTiming flag specified. This event need
 * be freed with topsEventDestroy. Operations on the imported event after the
 * exported event has been freed with topsEventDestroy will result in undefined
 * behavior. If the function is called within the same process where handle is
 * returned by topsIpcGetEventHandle, it will return topsErrorInvalidContext.
 *
 * Note. This function is only supported when the event is exported for a
 * remote process on a different device. If the event is exported for a remote
 * process on the same device, please use topsIpcOpenEventHandle instead.
 *
 * @param[out]  event  Pointer to topsEvent_t to return the event
 * @param[in]   handle The opaque interprocess handle to open
 * @param[in]   map The link that is expected to pass
 * @param[in]   port ESL port id
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorInvalidContext
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | No                       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsIpcOpenEventHandleExt(topsEvent_t* event,
                                   topsIpcEventHandle_t handle,
                                   topsTopologyMapType map,
                                   int port);

// end doxygen Device
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Error Error Handling
 *  @{
 *  This section describes the error handling functions of TOPS runtime API.
 */
/**
 * @brief Return last error returned by any TOPS runtime API call and resets the stored error code to
 * #topsSuccess
 *
 * @returns return code from last TOPS called from the active host thread
 *
 * Returns the last error that has been returned by any of the runtime calls in the same host
 * thread, and then resets the saved error to #topsSuccess.
 *
 * @see topsGetErrorString, topsGetLastError, topsPeekAtLastError, topsError_t
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaGetLastError         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsGetLastError(void);
/**
 * @brief Return last error returned by any TOPS runtime API call.
 *
 * @return #topsSuccess
 *
 * Returns the last error that has been returned by any of the runtime calls in the same host
 * thread. Unlike topsGetLastError, this function does not reset the saved error code.
 *
 * @see topsGetErrorString, topsGetLastError, topsPeekAtLastError, topsError_t
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaPeekAtLastError      |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsPeekAtLastError(void);
/**
 * @brief Return name of the specified error code in text form.
 *
 * @param tops_error Error code to convert to name.
 * @return const char pointer to the NULL-terminated error name
 *
 * @see topsGetErrorString, topsGetLastError, topsPeekAtLastError, topsError_t
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaGetErrorName         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
const char* topsGetErrorName(topsError_t tops_error);
/**
 * @brief Return handy text string message to explain the error which occurred
 *
 * @param topsError Error code to convert to string.
 * @return const char pointer to the NULL-terminated error string
 *
 * @warning : This function returns the name of the error (same as topsGetErrorName)
 *
 * @see topsGetErrorName, topsGetLastError, topsPeekAtLastError, topsError_t
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaGetErrorString       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
const char* topsGetErrorString(topsError_t topsError);
// end doxygen Error
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Stream Stream Management
 *  @{
 *  This section describes the stream management functions of TOPS runtime API.
 *  The following Stream APIs are not (yet) supported in TOPS:
 *  - topsStreamAttachMemAsync is a nop
 */

/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Pointer to new stream
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that can be used to
 * reference the newly created stream in subsequent topsStream* commands.  The stream is allocated on
 * the heap and will remain allocated even if the handle goes out-of-scope.  To release the memory
 * used by the stream, application must call topsStreamDestroy.
 *
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * @see topsStreamSynchronize, topsStreamWaitEvent, topsStreamDestroy
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaStreamCreate         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamCreate(topsStream_t* stream);
/**
 * @brief Create an asynchronous stream.
 *
 * @param[in, out] stream Pointer to new stream
 * @param[in ] flags to control stream creation.
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * Create a new asynchronous stream.  @p stream returns an opaque handle that
 * can be used to reference the newly created stream in subsequent topsStream*
 * commands.  The stream is allocated on the heap and will remain allocated even
 * if the handle goes out-of-scope.  To release the memory used by the stream,
 * application must call topsStreamDestroy. Flags controls behavior of the
 * stream.  See #topsStreamDefault, #topsStreamNonBlocking.
 *
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * @see topsStreamSynchronize, topsStreamWaitEvent, topsStreamDestroy
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaStreamCreateWithFlags|
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamCreateWithFlags(topsStream_t* stream, unsigned int flags);
/**
 * @brief Destroys the specified stream.
 *
 * @param[in] stream stream to destroy
 *
 * @return #topsSuccess #topsErrorInvalidHandle
 *
 * Destroys the specified stream.
 *
 * If commands are still executing on the specified stream, some may complete execution before the
 * queue is deleted.
 *
 * The queue may be destroyed while some commands are still inflight, or may wait for all commands
 * queued to the stream before destroying it.
 *
 * Note: stream resource should be released before process exit
 *
 * @see topsStreamCreate, topsStreamQuery, topsStreamWaitEvent,
 * topsStreamSynchronize
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaStreamDestroy        |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamDestroy(topsStream_t stream);
/**
 * @brief Return #topsSuccess if all of the operations in the specified @p stream have completed, or
 * #topsErrorNotReady if not.
 *
 * @param[in] stream stream to query
 *
 * @return #topsSuccess, #topsErrorNotReady, #topsErrorInvalidHandle
 *
 * This is thread-safe and returns a snapshot of the current state of the queue.  However, if other
 * host threads are sending work to the stream, the status may change immediately after the function
 * is called.  It is typically used for debug.
 *
 * @see topsStreamCreate, topsStreamWaitEvent, topsStreamSynchronize,
 * topsStreamDestroy
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaStreamQuery          |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamQuery(topsStream_t stream);
/**
 * @brief Wait for all commands in stream to complete.
 *
 * @param[in] stream stream identifier.
 *
 * @return #topsSuccess, #topsErrorInvalidHandle
 *
 * This command is host-synchronous : the host will block until the specified stream is empty.
 *
 * This command follows standard null-stream semantics.  Specifically, specifying the null stream
 * will cause the command to wait for other streams on the same device to complete all pending
 * operations.
 *
 * This command honors the topsDeviceLaunchBlocking flag, which controls whether the wait is active
 * or blocking.
 *
 * @see topsStreamCreate, topsStreamWaitEvent, topsStreamDestroy
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaStreamSynchronize    |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamSynchronize(topsStream_t stream);
/**
 * @brief Make the specified compute stream wait for an event
 *
 * @param[in] stream stream to make wait.
 * @param[in] event event to wait on
 * @param[in] flags control operation [must be 0]
 *
 * @return #topsSuccess, #topsErrorInvalidHandle
 *
 * This function inserts a wait operation into the specified stream.
 * All future work submitted to @p stream will wait until @p event reports completion before
 * beginning execution.
 *
 * This function only waits for commands in the current stream to complete.  Notably,, this function
 * does not implicit wait for commands in the default stream to complete, even if the specified
 * stream is created with topsStreamNonBlocking = 0.
 *
 * @see topsStreamCreate, topsStreamSynchronize, topsStreamDestroy
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaStreamWaitEvent      |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamWaitEvent(topsStream_t stream, topsEvent_t event, unsigned int flags);
/**
 * Stream CallBack struct
 */
typedef void (*topsStreamCallback_t)(topsStream_t stream, topsError_t status, void* userData);
/**
 * @brief Adds a callback to be called on the host after all currently enqueued
 * items in the stream have completed.  For each
 * topsStreamAddCallback call, a callback will be executed exactly once.
 * The callback will block later work in the stream until it is finished.
 * @param[in] stream   - Stream to add callback to
 * @param[in] callback - The function to call once preceding stream operations are complete
 * @param[in] userData - User specified data to be passed to the callback function
 * @param[in] flags    - topsStreamDefault: non-blocking stream execution;
 *                       topsStreamCallbackBlocking: stream blocks until callback is completed.
 * @return #topsSuccess, #topsErrorInvalidHandle, #topsErrorNotSupported
 *
 * @see topsStreamCreate, topsStreamQuery, topsStreamSynchronize,
 * topsStreamWaitEvent, topsStreamDestroy
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaStreamAddCallback    |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamAddCallback(topsStream_t stream, topsStreamCallback_t callback, void* userData,
                                unsigned int flags);

/**
 * @brief Write a value to local device memory.
 *
 * @param [in] dst  - The device address to write to.
 * @param [in] value - The value to write.
 * @param [in] flags - Reserved for future expansion; must be 0.
 *
 * @returns #topsSuccess, #topsErrorInvalidDevicePointer
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuStreamWriteValue32     |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamWriteValue32(topsDeviceptr_t dst, int value, unsigned int flags);

/**
 * @brief Write a value to local device memory async.
 *
 * @param [in] dst  - The device address to write to.
 * @param [in] value - The value to compare with the memory location.
 * @param [in] flags - Reserved for future expansion; must be 0.
 * @param [in] stream - The stream to synchronize on the memory location.
 *
 * @returns #topsSuccess, #topsErrorInvalidDevicePointer
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuStreamWriteValue32      |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamWriteValue32Async(topsDeviceptr_t dst, int value, unsigned int flags,
                                        topsStream_t stream __dparm(0));

/**
 * @brief Wait on a memory location.
 *
 * @param [in] dst  - The memory location to wait on.
 * @param [in] value - The value to compare with the memory location.
 * @param [in] flags - Reserved for future expansion; must be 0.
 *
 * @returns #topsSuccess, #topsErrorInvalidDevicePointer
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuStreamWaitValue32      |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamWaitValue32(topsDeviceptr_t dst, int value, unsigned int flags);

/**
 * @brief Wait on a memory location async.
 *
 * @param [in] dst  - The memory location to wait on.
 * @param [in] value - The value to compare with the memory location.
 * @param [in] flags - Reserved for future expansion; must be 0.
 * @param [in] stream - The stream to synchronize on the memory location.
 *
 * @returns #topsSuccess, #topsErrorInvalidDevicePointer
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuStreamWaitValue32      |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsStreamWaitValue32Async(topsDeviceptr_t dst, int value, unsigned int flags,
                                       topsStream_t stream __dparm(0));
// end doxygen Stream
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Event Event Management
 *  @{
 *  This section describes the event management functions of the TOPS runtime application programming interface.
 */
/**
 * @brief Create an event object with the specified flags.
 *
 * @param[in,out] event Returns the newly created event.
 * @param[in] flags Flags to control event behavior.
 *
 * @returns #topsSuccess, #topsErrorNotInitialized, #topsErrorInvalidValue, #topsErrorLaunchFailure, #topsErrorOutOfMemory
 *
 * Creates an event object for the current device with the specified flags. Valid values include:
 *  -#topsEventDefault: Default event create flag. The event will use active synchronization and will support timing.
 *  Blocking synchronization provides lowest possible latency at the expense of dedicating a CPU to poll on the event.
 *  -#topsEventBlockingSync: Specifies that event should use blocking synchronization. A host thread that uses
 *  topsEventSynchronize() to wait on an event created with this flag will block until the event actually completes.
 *  -#topsEventDisableTiming: Specifies that the created event does not need to record timing data. Events created with
 *  this flag specified and the topsEventBlockingSync flag not specified will provide the best performance when used with
 *  topsStreamWaitEvent() and topsEventQuery().
 *  -#topsEventInterprocess: Specifies that the created event may be used as an interprocess event by topsIpcGetEventHandle().
 *  topsEventInterprocess must be specified along with topsEventDisableTiming.
 *
 * @note
 *  + Note that this function may also return error codes from previous, asynchronous launches.
 *
 * @see topsEventCreate, topsEventSynchronize, topsEventDestroy, topsEventElapsedTime
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaEventCreateWithFlags |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsEventCreateWithFlags(topsEvent_t* event, unsigned flags);
/**
 *  Create an event object.
 *
 * @param[in,out] event Returns the newly created event.
 *
 * @returns #topsSuccess, #topsErrorNotInitialized, #topsErrorInvalidValue,
 * #topsErrorLaunchFailure, #topsErrorOutOfMemory
 *
 * Creates an event object for the current device using topsEventDefault.
 *
 * @see topsEventRecord, topsEventQuery, topsEventSynchronize,
 * topsEventDestroy, topsEventElapsedTime
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaEventCreate          |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsEventCreate(topsEvent_t* event);
/**
 * @brief Record an event in the specified stream.
 *
 * @param[in] event event to record.
 * @param[in] stream stream in which to record event.
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorNotInitialized,
 * #topsErrorInvalidHandle, #topsErrorLaunchFailure
 *
 * Captures in event the contents of stream at the time of this call. event and stream
 * must be on the same TOPS context. Calls such as topsEventQuery() or topsStreamWaitEvent()
 * will then examine or wait for completion of the work that was captured. Uses of stream after
 * this call do not modify event.
 *
 * topsEventRecord() can be called multiple times on the same event and will overwrite the
 * previously captured state. Other APIs such as topsStreamWaitEvent() use the most recently
 * captured state at the time of the API call, and are not affected by later calls to topsEventRecord().
 * Before the first call to topsEventRecord(), an event represents an empty set of work, so for example
 * topsEventQuery() would return topsSuccess.
 *
 * topsEventQuery() or topsEventSynchronize() must be used to determine when the event
 * transitions from "recording" (after topsEventRecord() is called) to "recorded"
 * (when timestamps are set, if requested).
 *
 * Events which are recorded in a non-NULL stream will transition to from recording to
 * "recorded" state when they reach the head of the specified stream, after all previous
 * commands in that stream have completed executing.
 *
 * If topsEventRecord() has been previously called on this event, then this call will overwrite any
 * existing state in event.
 *
 * If this function is called on an event that is currently being recorded, results are undefined
 * - either outstanding recording may save state into the event, and the order is not guaranteed.
 *
 * @see topsEventCreate, topsEventQuery, topsEventSynchronize,
 * topsEventDestroy, topsEventElapsedTime
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaEventRecord          |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KDM           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
#ifdef __cplusplus
topsError_t topsEventRecord(topsEvent_t event, topsStream_t stream = NULL);
#else
topsError_t topsEventRecord(topsEvent_t event, topsStream_t stream);
#endif
/**
 * @brief Destroy the specified event.
 *
 * @param[in] event Event to destroy.
 *
 * @returns #topsSuccess, #topsErrorNotInitialized, #topsErrorInvalidValue,
 * #topsErrorLaunchFailure
 *
 * Releases memory associated with the event. An event may be destroyed before
 * it is complete (i.e., while topsEventQuery() would return topsErrorNotReady).
 * If the event is recording but has not completed recording when topsEventDestroy()
 * is called, the function will return immediately and any associated resources will
 * automatically be released asynchronously at completion.
 *
 * @note Use of the handle after this call is undefined behavior.
 *
 * @see topsEventCreate, topsEventQuery, topsEventSynchronize, topsEventRecord,
 * topsEventElapsedTime
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaEventDestroy         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsEventDestroy(topsEvent_t event);
/**
 * @brief Wait for an event to complete.
 *
 * @param[in] event Event on which to wait.
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorNotInitialized,
 * #topsErrorInvalidHandle, #topsErrorLaunchFailure
 *
 * This function will block until the event is ready, waiting for all previous work in the stream
 * specified when event was recorded with topsEventRecord().
 *
 * If topsEventRecord() has not been called on @p event, this function returns immediately.
 *
 * Note:This function needs to support topsEventBlockingSync parameter.
 *
 * @see topsEventCreate, topsEventQuery, topsEventDestroy, topsEventRecord,
 * topsEventElapsedTime
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaEventSynchronize     |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsEventSynchronize(topsEvent_t event);
/**
 * @brief Return the elapsed time between two events.
 *
 * @param[out] ms : Return time between start and stop in ms.
 * @param[in]  start : Start event.
 * @param[in]  stop  : Stop event.
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorNotReady, #topsErrorInvalidHandle,
 * #topsErrorNotInitialized, #topsErrorLaunchFailure
 *
 * Computes the elapsed time between two events. Time is computed in ms, with
 * a resolution of approximately 1 us.
 *
 * Events which are recorded in a NULL stream will block until all commands
 * on all other streams complete execution, and then record the timestamp.
 *
 * Events which are recorded in a non-NULL stream will record their timestamp
 * when they reach the head of the specified stream, after all previous
 * commands in that stream have completed executing.  Thus the time that
 * the event recorded may be significantly after the host calls topsEventRecord().
 *
 * If topsEventRecord() has not been called on either event, then #topsErrorInvalidHandle is
 * returned. If topsEventRecord() has been called on both events, but the timestamp has not yet been
 * recorded on one or both events (that is, topsEventQuery() would return #topsErrorNotReady on at
 * least one of the events), then #topsErrorNotReady is returned.
 *
 * Note, for TOPS Events used in kernel dispatch using topsExtLaunchKernelGGL/topsExtLaunchKernel,
 * events passed in topsExtLaunchKernelGGL/topsExtLaunchKernel are not explicitly recorded and should
 * only be used to get elapsed time for that specific launch. In case events are used across
 * multiple dispatches, for example, start and stop events from different topsExtLaunchKernelGGL/
 * topsExtLaunchKernel calls, they will be treated as invalid unrecorded events, TOPS will throw
 * error "topsErrorInvalidHandle" from topsEventElapsedTime.
 *
 * @see topsEventCreate, topsEventQuery, topsEventDestroy, topsEventRecord,
 * topsEventSynchronize
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaEventElapsedTime     |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsEventElapsedTime(float* ms, topsEvent_t start, topsEvent_t stop);
/**
 * @brief Query event status
 *
 * @param[in] event Event to query.
 *
 * @returns #topsSuccess, #topsErrorNotReady, #topsErrorInvalidHandle, #topsErrorInvalidValue,
 * #topsErrorNotInitialized, #topsErrorLaunchFailure
 *
 * Query the status of the specified event.  This function will return #topsErrorNotReady if all
 * commands in the appropriate stream (specified to topsEventRecord()) have completed.  If that work
 * has not completed, or if topsEventRecord() was not called on the event, then #topsSuccess is
 * returned.
 *
 * @see topsEventCreate, topsEventRecord, topsEventDestroy,
 * topsEventSynchronize, topsEventElapsedTime
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaEventQuery           |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsEventQuery(topsEvent_t event);
// end doxygen Events
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Memory Memory Management
 *  @{
 *  This section describes the memory management functions of TOPS runtime API.
 */
/**
 * @brief Return attributes for the specified pointer
 *
 * @param [out]  attributes  attributes for the specified pointer
 * @param [in]   ptr         pointer to get attributes for
 *
 * @return #topsSuccess, #topsErrorInvalidDevice, #topsErrorInvalidValue
 *
 * @see topsPointerGetAttribute
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaPointerGetAttributes |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsPointerGetAttributes(topsPointerAttribute_t* attributes, const void* ptr);
/**
 * @brief Returns information about the specified pointer.
 *
 * @param [in, out] data     returned pointer attribute value
 * @param [in]      attribute attribute to query for
 * @param [in]      ptr      pointer to get attributes for
 *
 * @return #topsSuccess, #topsErrorInvalidDevice, #topsErrorInvalidValue
 *
 * @see topsPointerGetAttributes
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuPointerGetAttribute    |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsPointerGetAttribute(void* data, topsPointer_attribute attribute,
                                  topsDeviceptr_t ptr);
/**
 * @brief Returns information about the specified pointer.
 *
 * @param [in]  numAttributes   number of attributes to query for
 * @param [in]  attributes      attributes to query for
 * @param [in, out] data        a two-dimensional containing pointers to memory locations
 *                              where the result of each attribute query will be written to
 * @param [in]  ptr             pointer to get attributes for
 *
 * @return #topsSuccess, #topsErrorInvalidDevice, #topsErrorInvalidValue
 *
 * @see topsPointerGetAttribute
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuPointerGetAttributes   |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsDrvPointerGetAttributes(unsigned int numAttributes, topsPointer_attribute* attributes,
                                      void** data, topsDeviceptr_t ptr);
/**
 * @brief Allocate memory on the default accelerator
 *
 * @param[out] ptr Pointer to the allocated memory
 * @param[in]  size Requested memory size
 *
 * If size is 0, no memory is allocated, *ptr returns non-nullptr, and topsSuccess is returned.
 *
 * @return #topsSuccess, #topsErrorOutOfMemory, #topsErrorInvalidValue (bad context, null *ptr)
 *
 * @see topsFree, topsHostFree, topsHostMalloc
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMalloc               |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMalloc(void** ptr, size_t size);
/**
 * @brief convert device memory to efcodec memory handle
 *
 * @param[out] ptr Pointer to the allocated memory handle
 * @param[in]  dev_addr Requested memory device address
 * @param[in]  size Requested memory size
 *
 * If size is 0, no memory is allocated, *ptr returns non-nullptr, and topsSuccess is returned.
 *
 * @return #topsSuccess, #topsErrorOutOfMemory, #topsErrorInvalidValue (bad context, null *ptr)
 *
 * @see topsFree, topsHostFree, topsHostMalloc
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | -                        |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | No
 *  gcu210 (dorado) | No
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsExtCodecMemHandle(void** pointer, uint64_t dev_addr,
                    size_t size);
/**
 * @brief Allocate memory on the default accelerator
 *
 * @param[out] ptr Pointer to the allocated memory
 * @param[in]  sizeBytes Requested memory size
 * @param[in]  flags Type of memory allocation
 *              flags only support topsDeviceMallocDefault/topsMallocTopDown/
 *              topsMallocForbidMergeMove/topsMallocPreferHighSpeedMem
 *
 * If size is 0, no memory is allocated, *ptr returns non-nullptr, and topsSuccess is returned.
 *
 * @return #topsSuccess, #topsErrorOutOfMemory, #topsErrorInvalidValue (bad context, null *ptr)
 *
 * @see topsFree, topsHostFree, topsHostMalloc
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | No                       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags);
/**
 * @brief Allocate device accessible page locked host memory
 *
 * @param[out] ptr Pointer to the allocated host pinned memory
 * @param[in]  size Requested memory size
 * @param[in]  flags Type of host memory allocation
 *
 * If size is 0, no memory is allocated, *ptr returns nullptr, and topsSuccess is returned.
 *
 * @return #topsSuccess, #topsErrorOutOfMemory
 *
 * @see topsSetDeviceFlags, topsHostFree
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaHostAlloc            |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsHostMalloc(void** ptr, size_t size, unsigned int flags);
/**
 * @brief Get Device pointer from Host Pointer allocated through topsHostMalloc
 *
 * @param[out] devPtr Device Pointer mapped to passed host pointer
 * @param[in]  hostPtr Host Pointer allocated through topsHostMalloc
 * @param[in]  flags Flags to be passed for extension
 *
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorOutOfMemory
 *
 * @see topsSetDeviceFlags, topsHostMalloc
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaHostGetDevicePointer |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags);
/**
 * @brief Return flags associated with host pointer
 *
 * @param[out] flagsPtr Memory location to store flags
 * @param[in]  hostPtr Host Pointer allocated through topsHostMalloc
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * @see topsHostMalloc
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaHostGetFlags         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
/**
 * @brief Register host memory so it can be accessed from the current device.
 *
 * @param[out] hostPtr Pointer to host memory to be registered.
 * @param[in] sizeBytes size of the host memory
 * @param[in] flags.  See below.
 *
 * Flags:
 *  - #topsHostRegisterDefault   Memory is Mapped and Portable
 *  - #topsHostRegisterPortable  Memory is considered registered by all contexts.  TOPS only supports
 * one context so this is always assumed true.
 *  - #topsHostRegisterMapped    Map the allocation into the address space for the current device.
 * The device pointer can be obtained with #topsHostGetDevicePointer.
 *
 *
 * After registering the memory, use #topsHostGetDevicePointer to obtain the mapped device pointer.
 * On many systems, the mapped device pointer will have a different value than the mapped host
 * pointer.  Applications must use the device pointer in device code, and the host pointer in device
 * code.
 *
 * On some systems, registered memory is pinned.  On some systems, registered memory may not be
 * actually be pinned but uses OS or hardware facilities to all GCU access to the host memory.
 *
 * Developers are strongly encouraged to register memory blocks which are aligned to the host
 * cache-line size. (typically 64-bytes but can be obtains from the CPUID instruction).
 *
 * If registering non-aligned pointers, the application must take care when register pointers from
 * the same cache line on different devices.  TOPS's coarse-grained synchronization model does not
 * guarantee correct results if different devices write to different parts of the same cache block -
 * typically one of the writes will "win" and overwrite data from the other registered memory
 * region.
 *
 * @return #topsSuccess, #topsErrorOutOfMemory
 *
 * @see topsHostUnregister, topsHostGetFlags, topsHostGetDevicePointer
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaHostRegister         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
/**
 * @brief Un-register host pointer
 *
 * @param[in] hostPtr Host pointer previously registered with #topsHostRegister
 * @return Error code
 *
 * @see topsHostRegister
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaHostUnregister       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsHostUnregister(void* hostPtr);
/**
 * @brief Free memory allocated by the tops memory allocation API.
 * This API performs an implicit topsDeviceSynchronize() call.
 * If pointer is NULL, the tops runtime is initialized and topsSuccess is returned.
 *
 * @param[in] ptr Pointer to memory to be freed
 * @return #topsSuccess
 * @return #topsErrorInvalidDevicePointer (if pointer is invalid, including host pointers allocated
 * with topsHostMalloc)
 *
 * @see topsMalloc, topsHostFree, topsHostMalloc
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaFree                 |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsFree(void* ptr);
/**
 * @brief Free memory allocated by the tops host memory allocation API
 * This API performs an implicit topsDeviceSynchronize() call.
 * If pointer is NULL, the tops runtime is initialized and topsSuccess is returned.
 *
 * @param[in] ptr Pointer to memory to be freed
 * @return #topsSuccess,
 *         #topsErrorInvalidValue (if pointer is invalid, including device pointers allocated with
 * topsMalloc)
 *
 * @see topsMalloc, topsFree, topsHostMalloc
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaFreeHost             |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsHostFree(void* ptr);
/**
 * @brief Copy data from src to dst.
 *
 * It supports memory from host to device,
 * device to host, device to device and host to host
 * The src and dst must not overlap.
 *
 * For topsMemcpy, the copy is always performed by the current device (set by topsSetDevice).
 * For multi-gcu or peer-to-peer configurations, it is recommended to set the current device to the
 * device where the src data is physically located. For optimal peer-to-peer copies, the copy device
 * must be able to access the src and dst pointers (by calling topsDeviceEnablePeerAccess with copy
 * agent as the current device and src/dest as the peerDevice argument.  if this is not done, the
 * topsMemcpy will still work, but will perform the copy using a staging buffer on the host.
 * Calling topsMemcpy with dst and src pointers that do not match the topsMemcpyKind results in
 * undefined behavior.
 *
 * @param[out]  dst Data being copy to
 * @param[in]  src Data being copy from
 * @param[in]  sizeBytes Data size in bytes
 * @param[in]  kind Memory copy type
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryFree, #topsErrorUnknown
 *
 * @see topsMalloc, topsFree, topsHostMalloc, topsHostFree, topsMemGetAddressRange, topsMemGetInfo,
 * topsHostGetDevicePointer, topsMemcpyDtoD, topsMemcpyDtoDAsync, topsMemcpyDtoH, topsMemcpyDtoHAsync,
 * topsMemcpyHtoD, topsMemcpyHtoDAsync
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemcpy               |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpy(void* dst, const void* src, size_t sizeBytes, topsMemcpyKind kind);
/**
 * @brief Copy data from src to dst.
 *
 * It supports memory from host to device,
 * device to host, device to device and host to host
 * The src and dst must not overlap.
 *
 * @param[out]  dst Data being copy to
 * @param[in]  src Data being copy from
 * @param[in]  sizeBytes Data size in bytes
 * @param[in]  kind Memory copy type
 * @param[in]  stream Stream to enqueue this operation.
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryFree, #topsErrorUnknown
 *
 * @see topsMalloc, topsFree, topsHostMalloc, topsHostFree, topsMemGetAddressRange, topsMemGetInfo,
 * topsHostGetDevicePointer, topsMemcpyDtoD, topsMemcpyDtoDAsync, topsMemcpyDtoH, topsMemcpyDtoHAsync,
 * topsMemcpyHtoD, topsMemcpyHtoDAsync
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | No                       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,
                               topsMemcpyKind kind, topsStream_t stream);
/**
 * @brief Copy data from Host to Device
 *
 * @param[out]  dst Data being copy to
 * @param[in]   src Data being copy from
 * @param[in]   sizeBytes Data size in bytes
 *
 * @return #topsSuccess, #topsErrorDeInitialized, #topsErrorNotInitialized, #topsErrorInvalidContext,
 * #topsErrorInvalidValue
 *
 * @see topsMalloc, topsFree, topsHostMalloc, topsHostFree, topsMemGetAddressRange, topsMemGetInfo,
 * topsHostGetDevicePointer, topsMemcpyDtoD, topsMemcpyDtoDAsync, topsMemcpyDtoH, topsMemcpyDtoHAsync,
 * topsMemcpyHtoD, topsMemcpyHtoDAsync
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemcpyHtoD             |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyHtoD(topsDeviceptr_t dst, void* src, size_t sizeBytes);
/**
 * @brief Copy data from Device to Host
 *
 * @param[out]  dst Data being copy to
 * @param[in]   src Data being copy from
 * @param[in]   sizeBytes Data size in bytes
 *
 * @return #topsSuccess, #topsErrorDeInitialized, #topsErrorNotInitialized, #topsErrorInvalidContext,
 * #topsErrorInvalidValue
 *
 * @see topsMalloc, topsFree, topsHostMalloc, topsHostFree, topsMemGetAddressRange, topsMemGetInfo,
 * topsHostGetDevicePointer, topsMemcpyDtoD, topsMemcpyDtoDAsync, topsMemcpyDtoHAsync, topsMemcpyHtoD,
 * topsMemcpyHtoDAsync
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemcpyDtoH             |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyDtoH(void* dst, topsDeviceptr_t src, size_t sizeBytes);
/**
 * @brief Copy data from Device to Device
 *
 * @param[out]  dst Data being copy to
 * @param[in]   src Data being copy from
 * @param[in]   sizeBytes Data size in bytes
 *
 * @return #topsSuccess, #topsErrorDeInitialized, #topsErrorNotInitialized, #topsErrorInvalidContext,
 * #topsErrorInvalidValue
 *
 * @see topsMalloc, topsFree, topsHostMalloc, topsHostFree, topsMemGetAddressRange, topsMemGetInfo,
 * topsHostGetDevicePointer, topsMemcpyDtoDAsync, topsMemcpyDtoH, topsMemcpyDtoHAsync, topsMemcpyHtoD,
 * topsMemcpyHtoDAsync
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemcpyDtoD             |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyDtoD(topsDeviceptr_t dst, topsDeviceptr_t src, size_t sizeBytes);
/**
 * @brief Copy data from Host to Device asynchronously
 *
 * @param[out]  dst Data being copy to
 * @param[in]   src Data being copy from
 * @param[in]   sizeBytes Data size in bytes
 * @param[in]   stream Stream to enqueue this operation.
 *
 * @return #topsSuccess, #topsErrorDeInitialized, #topsErrorNotInitialized, #topsErrorInvalidContext,
 * #topsErrorInvalidValue
 *
 * @see topsMalloc, topsFree, topsHostMalloc, topsHostFree, topsMemGetAddressRange, topsMemGetInfo,
 * topsHostGetDevicePointer, topsMemcpyDtoD, topsMemcpyDtoDAsync, topsMemcpyDtoH, topsMemcpyDtoHAsync,
 * topsMemcpyHtoD
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemcpyHtoDAsync        |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyHtoDAsync(topsDeviceptr_t dst, void* src, size_t sizeBytes, topsStream_t stream);
/**
 * @brief Copy data from Device to Host asynchronously
 *
 * @param[out]  dst Data being copy to
 * @param[in]   src Data being copy from
 * @param[in]   sizeBytes Data size in bytes
 * @param[in]   stream Stream to enqueue this operation.
 *
 * @return #topsSuccess, #topsErrorDeInitialized, #topsErrorNotInitialized, #topsErrorInvalidContext,
 * #topsErrorInvalidValue
 *
 * @see topsMalloc, topsFree, topsHostMalloc, topsHostFree, topsMemGetAddressRange, topsMemGetInfo,
 * topsHostGetDevicePointer, topsMemcpyDtoD, topsMemcpyDtoDAsync, topsMemcpyDtoH, topsMemcpyHtoD,
 * topsMemcpyHtoDAsync
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemcpyDtoHAsync        |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyDtoHAsync(void* dst, topsDeviceptr_t src, size_t sizeBytes, topsStream_t stream);
/**
 * @brief Copy data from Device to Device asynchronously
 *
 * @param[out]  dst Data being copy to
 * @param[in]   src Data being copy from
 * @param[in]   sizeBytes Data size in bytes
 * @param[in]   stream Stream to enqueue this operation.
 *
 * @return #topsSuccess, #topsErrorDeInitialized, #topsErrorNotInitialized, #topsErrorInvalidContext,
 * #topsErrorInvalidValue
 *
 * @see topsMalloc, topsFree, topsHostMalloc, topsHostFree, topsMemGetAddressRange, topsMemGetInfo,
 * topsHostGetDevicePointer, topsMemcpyDtoD, topsMemcpyDtoH, topsMemcpyDtoHAsync, topsMemcpyHtoD,
 * topsMemcpyHtoDAsync
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemcpyDtoDAsync        |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyDtoDAsync(topsDeviceptr_t dst, topsDeviceptr_t src, size_t sizeBytes,
                              topsStream_t stream);

/**
 * @brief Returns a global pointer from a module.
 * Returns in *dptr and *bytes the pointer and size of the global symbol located in module hmod.
 * If no variable of that name exists, it returns topsErrorNotFound. Both parameters dptr and bytes are optional.
 * If one of them is NULL, it is ignored and topsSuccess is returned.
 *
 * @param[out]  dptr  Returns global device pointer
 * @param[out]  bytes Returns global size in bytes
 * @param[in]   hmod  Module to retrieve global from
 * @param[in]   name  Name of global to retrieve
 *
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorNotFound, #topsErrorInvalidContext
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuModuleGetGlobal        |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsModuleGetGlobal(topsDeviceptr_t* dptr, size_t* bytes,
    topsModule_t hmod, const char* name);

/**
 * @brief Gets device pointer associated with symbol on the device.
 *
 * @param[out]  devPtr  pointer to the device associated the symbol
 * @param[in]   symbol  pointer to the symbol of the device
 *
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaGetSymbolAddress     |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsGetSymbolAddress(void** devPtr, const void* symbol);

/**
 * @brief Gets the size of the given symbol on the device.
 *
 * @param[in]   symbol  pointer to the device symbol
 * @param[out]  size  pointer to the size
 *
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaGetSymbolSize        |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsGetSymbolSize(size_t* size, const void* symbol);

/**
 * @brief Copies data to the given symbol on the device.
 * Symbol TOPS APIs allow a kernel to define a device-side data symbol which can be accessed on
 * the host side. The symbol can be in __constant or device space.
 * Note that the symbol name needs to be encased in the TOPS_SYMBOL macro.
 * This also applies to topsMemcpyFromSymbol, topsGetSymbolAddress, and topsGetSymbolSize.
 *
 * @param[out]  symbol  pointer to the device symbol
 * @param[in]   src  pointer to the source address
 * @param[in]   sizeBytes  size in bytes to copy
 * @param[in]   offset  offset in bytes from start of symbol
 * @param[in]   kind  type of memory transfer
 *
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemcpyToSymbol       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyToSymbol(const void* symbol, const void* src,
                             size_t sizeBytes, size_t offset __dparm(0),
                             topsMemcpyKind kind __dparm(topsMemcpyHostToDevice));

/**
 * @brief Copies data to the given symbol on the device asynchronously.
 *
 * @param[out]  symbol  pointer to the device symbol
 * @param[in]   src  pointer to the source address
 * @param[in]   sizeBytes  size in bytes to copy
 * @param[in]   offset  offset in bytes from start of symbol
 * @param[in]   kind  type of memory transfer
 * @param[in]   stream  stream identifier
 *
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemcpyToSymbolAsync  |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyToSymbolAsync(const void* symbol, const void* src,
                                  size_t sizeBytes, size_t offset,
                                  topsMemcpyKind kind, topsStream_t stream __dparm(0));

/**
 * @brief Copies data from the given symbol on the device.
 *
 * @param[out]  dptr  Returns pointer to destination memory address
 * @param[in]   symbol  pointer to the symbol address on the device
 * @param[in]   sizeBytes  size in bytes to copy
 * @param[in]   offset  offset in bytes from the start of symbol
 * @param[in]   kind  type of memory transfer
 *
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemcpyFromSymbol     |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyFromSymbol(void* dst, const void* symbol,
                               size_t sizeBytes, size_t offset __dparm(0),
                               topsMemcpyKind kind __dparm(topsMemcpyDeviceToHost));

/**
 * @brief Copies data from the given symbol on the device asynchronously.
 *
 * @param[out]  dptr  Returns pointer to destination memory address
 * @param[in]   symbol  pointer to the symbol address on the device
 * @param[in]   sizeBytes  size in bytes to copy
 * @param[in]   offset  offset in bytes from the start of symbol
 * @param[in]   kind  type of memory transfer
 * @param[in]   stream  stream identifier
 *
 * @return #topsSuccess, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemcpyFromSymbolAsync|
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyFromSymbolAsync(void* dst, const void* symbol,
                                    size_t sizeBytes, size_t offset,
                                    topsMemcpyKind kind,
                                    topsStream_t stream __dparm(0));
/**
 * @brief Copy data from src to dst asynchronously.
 *
 * @warning If host or dest are not pinned, the memory copy will be performed synchronously.  For
 * best performance, use topsHostMalloc to allocate host memory that is transferred asynchronously.
 *
 * @warning topsMemcpyAsync does not support overlapped H2D and D2H copies.
 * For topsMemcpy, the copy is always performed by the device associated with the specified stream.
 *
 * For multi-gcu or peer-to-peer configurations, it is recommended to use a stream which is a
 * attached to the device where the src data is physically located. For optimal peer-to-peer copies,
 * the copy device must be able to access the src and dst pointers (by calling
 * topsDeviceEnablePeerAccess with copy agent as the current device and src/dest as the peerDevice
 * argument.  if this is not done, the topsMemcpy will still work, but will perform the copy using a
 * staging buffer on the host.
 *
 * @param[out] dst Data being copy to
 * @param[in]  src Data being copy from
 * @param[in]  sizeBytes Data size in bytes
 * @param[in]  kind  type of memory transfer
 * @param[in]  stream  stream identifier
 *
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryFree, #topsErrorUnknown
 *
 * @see topsMalloc, topsFree, topsHostMalloc, topsHostFree, topsMemGetAddressRange, topsMemGetInfo,
 * topsHostGetDevicePointer, topsMemcpyDtoD, topsMemcpyDtoH, topsMemcpyDtoHAsync, topsMemcpyHtoD,
 * topsMemcpyHtoDAsync
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemcpyAsync          |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemcpyAsync(void* dst, const void* src, size_t sizeBytes, topsMemcpyKind kind,
                          topsStream_t stream __dparm(0));
/**
 * @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * byte value value.
 *
 * @param[out] dst Dst Data being filled
 * @param[in]  value Constant value to be set
 * @param[in]  sizeBytes Data size in bytes
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorNotInitialized
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemset               |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemset(void* dst, int value, size_t sizeBytes);
/**
 * @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * byte value value.
 *
 * @param[out] dst Data ptr to be filled
 * @param[in]  value Constant value to be set
 * @param[in]  count Number of values to be set
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorNotInitialized
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemsetD8               |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemsetD8(topsDeviceptr_t dest, unsigned char value, size_t count);
/**
 * @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * byte value value.
 *
 * topsMemsetD8Async() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 * @param[out] dest Data ptr to be filled
 * @param[in]  value Constant value to be set
 * @param[in]  count Number of values to be set
 * @param[in]  stream - Stream identifier
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorNotInitialized
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemsetD8Async          |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemsetD8Async(topsDeviceptr_t dest, unsigned char value, size_t count, topsStream_t stream __dparm(0));
/**
 * @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * short value value.
 *
 * @param[out] dest Data ptr to be filled
 * @param[in]  value Constant value to be set
 * @param[in]  count Number of values to be set
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorNotInitialized
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          |  cuMemsetD16             |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemsetD16(topsDeviceptr_t dest, unsigned short value, size_t count);
/**
 * @brief Fills the first sizeBytes bytes of the memory area pointed to by dest with the constant
 * short value value.
 *
 * topsMemsetD16Async() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 * @param[out] dest Data ptr to be filled
 * @param[in]  value Constant value to be set
 * @param[in]  count Number of values to be set
 * @param[in]  stream - Stream identifier
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorNotInitialized
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemsetD16Async         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemsetD16Async(topsDeviceptr_t dest, unsigned short value, size_t count, topsStream_t stream __dparm(0));
/**
 * @brief Fills the memory area pointed to by dest with the constant integer
 * value for specified number of times.
 *
 * @param[out] dest Data being filled
 * @param[in]  value Constant value to be set
 * @param[in]  count Number of values to be set
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorNotInitialized
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemsetD32              |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemsetD32(topsDeviceptr_t dest, int value, size_t count);
/**
 * @brief Fills the first sizeBytes bytes of the memory area pointed to by dev with the constant
 * byte value value.
 *
 * topsMemsetAsync() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 * @param[out] dst Pointer to device memory
 * @param[in]  value - Value to set for each byte of specified memory
 * @param[in]  sizeBytes - Size in bytes to set
 * @param[in]  stream - Stream identifier
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryFree
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemsetAsync         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemsetAsync(void* dst, int value, size_t sizeBytes, topsStream_t stream __dparm(0));
/**
 * @brief Fills the memory area pointed to by dev with the constant integer
 * value for specified number of times.
 *
 * topsMemsetD32Async() is asynchronous with respect to the host, so the call may return before the
 * memset is complete. The operation can optionally be associated to a stream by passing a non-zero
 * stream argument. If stream is non-zero, the operation may overlap with operations in other
 * streams.
 *
 * @param[out] dst Pointer to device memory
 * @param[in]  value - Value to set for each byte of specified memory
 * @param[in]  count - number of values to be set
 * @param[in]  stream - Stream identifier
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryFree
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuMemsetD32Async         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemsetD32Async(topsDeviceptr_t dst, int value, size_t count,
                             topsStream_t stream __dparm(0));
/**
 * @brief Query memory info.

 * Return snapshot of free memory, and total allocatable memory on the device.
 *
 * Returns in *free a snapshot of the current free memory.
 * @returns #topsSuccess, #topsErrorInvalidDevice, #topsErrorInvalidValue
 * @warning The free memory only accounts for memory allocated by this process and may be
 *optimistic.
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemGetInfo           |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 **/
topsError_t topsMemGetInfo(size_t* free, size_t* total);
/**
 * @brief Query memory pointer info.
 * Return size of the memory pointer.
 *
 * @param[out] size The size of memory pointer.
 * @param[in]  ptr Pointer to memory for query.
 * @returns #topsSuccess, #topsErrorInvalidDevice, #topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | No                       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 **/
topsError_t topsMemPtrGetInfo(void* ptr, size_t* size);

/**
 * @brief Get information on memory allocations.
 *
 * @param [out] pbase - Base pointer address
 * @param [out] psize - Size of allocation
 * @param [in]  dptr- Device Pointer
 *
 * @returns #topsSuccess, #topsErrorInvalidDevicePointer
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | No                       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsMemGetAddressRange(topsDeviceptr_t* pbase, size_t* psize, topsDeviceptr_t dptr);
// doxygen end Memory
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup PeerToPeer PeerToPeer Device Memory Access
 *  @{
 *  @warning PeerToPeer support is experimental.
 *  This section describes the PeerToPeer device memory access functions of TOPS runtime API.
 */
/**
 * @brief Checks if peer/esl access between two devices is possible.
 *
 * @param [in] deviceId - device id
 * @param [in] peerDeviceId - peer device id
 * @param [out] canAccess - access between two devices,
 *              bit[7~0]   : Each bit indicating corresponding port status: 1 link, 0 no-link.
 *              bit[15~8]  : p2p link type: 0 no-p2p-link, 1 PCIe switch link, 2 RCs link.
 *              bit[23~16] :cluster as device type: 1 cluster as device.
 *
 * @returns topsSuccess, topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaDeviceCanAccessPeer  |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 **/
topsError_t topsDeviceCanAccessPeer(int* canAccess, int deviceId, int peerDeviceId);
/**
 * @brief Set peer/esl access property.
 *
 * @param [in] peerDeviceId - peer device id
 * @param [in] flags - access property
 *
 * @returns topsSuccess, topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                  |
 * ------------------ | ------------------------  |
 * Interface          | cudaDeviceEnablePeerAccess|
 * Benchmark          | -                         |
 * First Shown Ver.   | -                         |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 **/
topsError_t topsDeviceEnablePeerAccess(int peerDeviceId, unsigned int flags);
/**
 * @brief Set peer access property for peer device's region.
 *
 * @param [in] peerDeviceId - peer device id
 * @param [in] peerDevPtr - the start device ptr of the peer device's region
 * @param [in] size - the size of the peer device's region
 * @param [out] devPtr - the p2p mapped device address
 *
 * @returns topsSuccess, topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                  |
 * ------------------ | ------------------------  |
 * Interface          | No                        |
 * Benchmark          | -                         |
 * First Shown Ver.   | -                         |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | YES
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| NO
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.2.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 **/
topsError_t topsDeviceEnablePeerAccessRegion(int peerDeviceId, void *peerDevPtr, size_t size, void **devPtr);
/**
 * @brief destroy peer access property for peer device's region.
 *
 * @param [in] peerDeviceId - peer device id
 * @param [in] peerDevPtr - the start device ptr of the peer device's region
 * @param [in] size - the size of the peer device's region
 *
 * @returns topsSuccess, topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                  |
 * ------------------ | ------------------------  |
 * Interface          | No                        |
 * Benchmark          | -                         |
 * First Shown Ver.   | -                         |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | YES
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| NO
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.2.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 **/
topsError_t topsDeviceDisablePeerAccessRegion(int peerDeviceId, void *peerDevPtr, size_t size);
/**
 * @brief Copies memory from one device to memory on another device.
 *
 * For topsMemcpyPeer, the copy is always performed by the current device (set by topsSetDevice).
 * For multi-gcu or peer-to-peer configurations, it is recommended to set the current device to the
 * device where the src data is physically located. For optimal peer-to-peer copies, the copy device
 * must be able to access the src and dst pointers (by calling topsDeviceEnablePeerAccess with copy
 * agent as the current device and src/dest as the peerDevice argument.
 *
 * @param[out] dst Data being copy to
 * @param[in]  dstDevice Dst device id
 * @param[in]  src Data being copy from
 * @param[in]  srcDevice Src device id
 * @param[in]  sizeBytes Data size in bytes
 *
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryFree, #topsErrorUnknown
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemcpyPeer           |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 **/
topsError_t topsMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
                           size_t sizeBytes);
/**
 * @brief Copies memory from one device to memory on another device asynchronously.
 *
 * For multi-gcu or peer-to-peer configurations, it is recommended to use a stream which is a
 * attached to the device where the src data is physically located. For optimal peer-to-peer copies,
 * the copy device must be able to access the src and dst pointers (by calling
 * topsDeviceEnablePeerAccess with copy agent as the current device and src/dest as the peerDevice
 * argument.
 *
 * @param[out] dst Data being copy to
 * @param[in]  dstDevice Dst device id
 * @param[in]  src Data being copy from
 * @param[in]  srcDevice Src device id
 * @param[in]  sizeBytes Data size in bytes
 * @param[in]  stream Stream identifier
 *
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryFree, #topsErrorUnknown
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaMemcpyPeerAsync      |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 **/

topsError_t topsMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice,
                                size_t sizeBytes, topsStream_t stream);

/**
 * @brief Copies memory from one device to memory on another device with special priority.
 *
 * For topsMemcpyPeerExt, the copy is always performed by the current device (set by topsSetDevice).
 * For multi-gcu or peer-to-peer configurations, it is recommended to set the current device to the
 * device where the src data is physically located. For optimal peer-to-peer copies, the copy device
 * must be able to access the src and dst pointers (by calling topsDeviceEnablePeerAccess with copy
 * agent as the current device and src/dest as the peerDevice argument.
 *
 * @param[out] dst Data being copy to
 * @param[in]  dstDevice Dst device id
 * @param[in]  src Data being copy from
 * @param[in]  srcDevice Src device id
 * @param[in]  sizeBytes Data size in bytes
 * @param[in]  map The link that is expected to pass
 * @param[in]  port ESL port id
 *
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryFree, #topsErrorUnknown
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | No                       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 **/
topsError_t topsMemcpyPeerExt(void* dst, int dstDevice, const void* src, int srcDevice,
                              size_t sizeBytes, topsTopologyMapType map, int port);

/**
 * @brief Copies memory from one device to memory on another device asynchronously with special priority.
 *
 * For topsMemcpyPeerExtAsync, the copy is always performed by the current device (set by topsSetDevice).
 * For multi-gcu or peer-to-peer configurations, it is recommended to set the current device to the
 * device where the src data is physically located. For optimal peer-to-peer copies, the copy device
 * must be able to access the src and dst pointers (by calling topsDeviceEnablePeerAccess with copy
 * agent as the current device and src/dest as the peerDevice argument.
 *
 * @param[out] dst Data being copy to
 * @param[in]  dstDevice Dst device id
 * @param[in]  src Data being copy from
 * @param[in]  srcDevice Src device id
 * @param[in]  sizeBytes Data size in bytes
 * @param[in]  map The link that is expected to pass
 * @param[in]  port ESL port id
 * @param[in]  stream Stream identifier
 *
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryFree, #topsErrorUnknown
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | No                       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 **/
topsError_t topsMemcpyPeerExtAsync(void* dst, int dstDevice, const void* src, int srcDevice,
                                size_t sizeBytes, topsTopologyMapType map, int port,
                                topsStream_t stream);

// doxygen end PeerToPeer
/**
 * @}
 */
/**
 *
 *  @defgroup Module Module Management
 *  @{
 *  This section describes the module management functions of TOPS runtime API.
 *
 */
/**
 * @brief Loads code object from file into a topsModule_t
 *
 * @param [in] fname
 * @param [out] module
 *
 * @returns topsSuccess, topsErrorInvalidValue, topsErrorInvalidContext, topsErrorFileNotFound,
 * topsErrorOutOfMemory, topsErrorSharedObjectInitFailed, topsErrorNotInitialized
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuModuleLoad             |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 *
 */
topsError_t topsModuleLoad(topsModule_t* module, const char* fname);
/**
 * @brief Frees the module
 *
 * @param [in] module
 *
 * @returns topsSuccess, topsErrorInvalidValue
 * module is freed and the code objects associated with it are destroyed
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuModuleUnload           |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsModuleUnload(topsModule_t module);
/**
 * @brief Function with kname will be extracted if present in module
 *
 * @param [in] module
 * @param [in] kname
 * @param [out] function
 *
 * @returns topsSuccess, topsErrorInvalidValue, topsErrorInvalidContext, topsErrorNotInitialized,
 * topsErrorNotFound,
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuModuleGetFunction      |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsModuleGetFunction(topsFunction_t* function, topsModule_t module, const char* kname);
/**
 * @brief Find out attributes for a given function.
 *
 * @param [out] attr
 * @param [in] func
 *
 * @returns topsSuccess, topsErrorInvalidValue, topsErrorInvalidDeviceFunction
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          |	cudaFuncGetAttributes    |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsFuncGetAttributes(struct topsFuncAttributes* attr, const void* func);
/**
 * @brief Find out a specific attribute for a given function.
 *
 * @param [out] value
 * @param [in]  attrib
 * @param [in]  hfunc
 *
 * @returns topsSuccess, topsErrorInvalidValue, topsErrorInvalidDeviceFunction
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | No                       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsFuncGetAttribute(int* value, topsFunction_attribute attrib, topsFunction_t hfunc);
/**
 * @brief builds module from code object which resides in host memory. Image is pointer to that
 * location.
 *
 * @param [in] image
 * @param [out] module
 *
 * @returns topsSuccess, topsErrorNotInitialized, topsErrorOutOfMemory, topsErrorNotInitialized
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuModuleLoadData         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsModuleLoadData(topsModule_t* module, const void* image);
/**
 * @brief builds module from code object which resides in host memory. Image is pointer to that
 * location. Options are not used. topsModuleLoadData is called.
 *
 * @param [in] image
 * @param [out] module
 * @param [in] numOptions Number of options
 * @param [in] options Options for JIT
 * @param [in] optionValues Option values for JIT
 *
 * @returns topsSuccess, topsErrorNotInitialized, topsErrorOutOfMemory, topsErrorNotInitialized
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuModuleLoadDataEx       |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsModuleLoadDataEx(topsModule_t* module, const void* image, unsigned int numOptions,
                               topsJitOption* options, void** optionValues);
/**
 * @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
 * to kernelparams or extra
 *
 * @param [in] f         Kernel to launch.
 * @param [in] gridDimX  X grid dimension specified as multiple of blockDimX.
 * @param [in] gridDimY  Y grid dimension specified as multiple of blockDimY.
 * @param [in] gridDimZ  Z grid dimension specified as multiple of blockDimZ.
 * @param [in] blockDimX X block dimensions specified in work-items
 * @param [in] blockDimY Y grid dimension specified in work-items
 * @param [in] blockDimZ Z grid dimension specified in work-items
 * @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
 * TOPS-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 * @param [in] kernelParams
 * @param [in] extra     Pointer to kernel arguments.   These are passed directly to the kernel and
 * must be in the memory layout and alignment expected by the kernel.
 *
 * Please note, TOPS does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32. So gridDim.x * blockDim.x, gridDim.y * blockDim.y
 * and gridDim.z * blockDim.z are always less than 2^32.
 *
 * @returns topsSuccess, topsInvalidDevice, topsErrorNotInitialized, topsErrorInvalidValue
 *
 * @warning kernellParams argument is not yet implemented in TOPS. Please use extra instead. Please
 * refer to tops_porting_driver_api.md for sample usage.
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cuLaunchKernel           |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsModuleLaunchKernel(topsFunction_t f, unsigned int gridDimX, unsigned int gridDimY,
                                 unsigned int gridDimZ, unsigned int blockDimX,
                                 unsigned int blockDimY, unsigned int blockDimZ,
                                 unsigned int sharedMemBytes, topsStream_t stream,
                                 void** kernelParams, void** extra);
/**
 * @brief launches kernel f with launch parameters and shared memory on stream with arguments passed
 * to kernelparams or extra, where thread blocks can cooperate and synchronize as they execute
 *
 * @param [in] f         Kernel to launch.
 * @param [in] gridDim   Grid dimensions specified as multiple of blockDim.
 * @param [in] blockDim  Block dimensions specified in work-items
 * @param [in] kernelParams A list of kernel arguments
 * @param [in] sharedMemBytes Amount of dynamic shared memory to allocate for this kernel. The
 * TOPS-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 *
 * Please note, TOPS does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns topsSuccess, topsInvalidDevice, topsErrorNotInitialized, topsErrorInvalidValue, topsErrorCooperativeLaunchTooLarge
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                   |
 * ------------------ | ------------------------   |
 * Interface          | cudaLaunchCooperativeKernel|
 * Benchmark          | -                          |
 * First Shown Ver.   | -                          |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsLaunchCooperativeKernel(const void* f, dim3 gridDim, dim3 blockDimX,
                                      void** kernelParams, size_t sharedMemBytes,
                                      topsStream_t stream);
// doxygen end Module
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Clang Launch API to support the triple-chevron syntax
 *  @{
 *  This section describes the API to support the triple-chevron syntax.
 */
/**
 * @brief Configure a kernel launch.
 *
 * @param [in] gridDim   grid dimension specified as multiple of blockDim.
 * @param [in] blockDim  block dimensions specified in work-items
 * @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
 * TOPS-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 *
 * Please note, TOPS does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns topsSuccess, topsInvalidDevice, topsErrorNotInitialized, topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaConfigureCall        |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsConfigureCall(dim3 gridDim, dim3 blockDim, size_t sharedMem __dparm(0), topsStream_t stream __dparm(0));
/**
 * @brief Set a kernel argument.
 *
 * @returns topsSuccess, topsInvalidDevice, topsErrorNotInitialized, topsErrorInvalidValue
 *
 * @param [in] arg    Pointer the argument in host memory.
 * @param [in] size   Size of the argument.
 * @param [in] offset Offset of the argument on the argument stack.
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaSetupArgument        |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsSetupArgument(const void* arg, size_t size, size_t offset);
/**
 * @brief Launch a kernel.
 *
 * @param [in] func Kernel to launch.
 *
 * @returns topsSuccess, topsInvalidDevice, topsErrorNotInitialized, topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaLaunch               |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsLaunchByPtr(const void* func);
/**
 * @brief Push configuration of a kernel launch.
 *
 * @param [in] gridDim   grid dimension specified as multiple of blockDim.
 * @param [in] blockDim  block dimensions specified in work-items
 * @param [in] sharedMem Amount of dynamic shared memory to allocate for this kernel. The
 * TOPS-Clang compiler provides support for extern shared declarations.
 * @param [in] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 *
 * Please note, TOPS does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns topsSuccess, topsInvalidDevice, topsErrorNotInitialized, topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                   |
 * ------------------ | ------------------------   |
 * Interface          | __cudaPushCallConfiguration|
 * Benchmark          | -                          |
 * First Shown Ver.   | -                          |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t __topsPushCallConfiguration(dim3 gridDim,
                                      dim3 blockDim,
                                      size_t sharedMem __dparm(0),
                                      topsStream_t stream __dparm(0));
/**
 * @brief Pop configuration of a kernel launch.
 *
 * @param [out] gridDim   grid dimension specified as multiple of blockDim.
 * @param [out] blockDim  block dimensions specified in work-items
 * @param [out] sharedMem Amount of dynamic shared memory to allocate for this kernel.  The
 * TOPS-Clang compiler provides support for extern shared declarations.
 * @param [out] stream    Stream where the kernel should be dispatched.  May be 0, in which case the
 * default stream is used with associated synchronization rules.
 *
 * Please note, TOPS does not support kernel launch with total work items defined in dimension with
 * size gridDim x blockDim >= 2^32.
 *
 * @returns topsSuccess, topsInvalidDevice, topsErrorNotInitialized, topsErrorInvalidValue
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                  |
 * ------------------ | ------------------------  |
 * Interface          | __cudaPopCallConfiguration|
 * Benchmark          | -                         |
 * First Shown Ver.   | -                         |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t __topsPopCallConfiguration(dim3 *gridDim,
                                     dim3 *blockDim,
                                     size_t *sharedMem,
                                     topsStream_t *stream);
/**
 * @brief C compliant kernel launch API
 *
 * @param [in] function_address - kernel stub function pointer.
 * @param [in] numBlocks - number of blocks
 * @param [in] dimBlocks - dimension of a block
 * @param [in] args - kernel arguments
 * @param [in] sharedMemBytes - Amount of dynamic shared memory to allocate for this kernel. The
 * TOPS-Clang compiler provides support for extern shared declarations.
 * @param [in] stream - Stream where the kernel should be dispatched.  May be 0, in which case the
 *  default stream is used with associated synchronization rules.
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, topsInvalidDevice
 *
 * @cond INTERNAL
 *
 * ####Detail Design#####
 *
 *
 * ####Competitive Analysis#####
 *   -                | Cuda API                 |
 * ------------------ | ------------------------ |
 * Interface          | cudaLaunchKernel         |
 * Benchmark          | -                        |
 * First Shown Ver.   | -                        |
 *
 * ####Compatibility Analysis#####
 *   hardware       | compatible
 * -------------    | -------------
 *  gcu100 (leo)    | NO
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.0
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
topsError_t topsLaunchKernel(const void* function_address,
                           dim3 numBlocks,
                           dim3 dimBlocks,
                           void** args,
                           size_t sharedMemBytes __dparm(0),
                           topsStream_t stream __dparm(0));
// doxygen end Clang launch
/**
 * @}
 */
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Graph Graph Management
 *  @{
 *  This section describes the graph management types & functions of TOPS runtime API.
 */

/**
 * An opaque value that represents a tops graph
 */
typedef struct itopsGraph* topsGraph_t;
/**
 * An opaque value that represents a tops graph node
 */
typedef struct topsGraphNode* topsGraphNode_t;
/**
 * An opaque value that represents a tops graph Exec
 */
typedef struct topsGraphExec* topsGraphExec_t;

/**
 * @brief topsGraphNodeType
 * @enum
 *
 */
typedef enum topsGraphNodeType {
  topsGraphNodeTypeKernel = 1,             ///< GCU kernel node
  topsGraphNodeTypeMemcpy = 2,             ///< Memcpy 3D node
  topsGraphNodeTypeMemset = 3,             ///< Memset 1D node
  topsGraphNodeTypeHost = 4,               ///< Host (executable) node
  topsGraphNodeTypeGraph = 5,              ///< Node which executes an embedded graph
  topsGraphNodeTypeEmpty = 6,              ///< Empty (no-op) node
  topsGraphNodeTypeWaitEvent = 7,          ///< External event wait node
  topsGraphNodeTypeEventRecord = 8,        ///< External event record node
  topsGraphNodeTypeMemcpy1D = 9,           ///< Memcpy 1D node
  topsGraphNodeTypeMemcpyFromSymbol = 10,  ///< MemcpyFromSymbol node
  topsGraphNodeTypeMemcpyToSymbol = 11,    ///< MemcpyToSymbol node
  topsGraphNodeTypeCount
} topsGraphNodeType;

typedef void (*topsHostFn_t)(void* userData);
typedef struct topsHostNodeParams {
  topsHostFn_t fn;
  void* userData;
} topsHostNodeParams;
typedef struct topsKernelNodeParams {
  dim3 blockDim;
  void** extra;
  void* func;
  dim3 gridDim;
  void** kernelParams;
  unsigned int sharedMemBytes;
} topsKernelNodeParams;
typedef struct topsMemsetParams {
  void* dst;
  unsigned int elementSize;
  size_t height;
  size_t pitch;
  unsigned int value;
  size_t width;
} topsMemsetParams;

/**
 * @brief topsGraphExecUpdateResult
 * @enum
 *
 */
typedef enum topsGraphExecUpdateResult {
  topsGraphExecUpdateSuccess = 0x0,  ///< The update succeeded
  topsGraphExecUpdateError = 0x1,  ///< The update failed for an unexpected reason which is described
                                  ///< in the return value of the function
  topsGraphExecUpdateErrorTopologyChanged = 0x2,  ///< The update failed because the topology changed
  topsGraphExecUpdateErrorNodeTypeChanged = 0x3,  ///< The update failed because a node type changed
  topsGraphExecUpdateErrorFunctionChanged =
      0x4,  ///< The update failed because the function of a kernel node changed
  topsGraphExecUpdateErrorParametersChanged =
      0x5,  ///< The update failed because the parameters changed in a way that is not supported
  topsGraphExecUpdateErrorNotSupported =
      0x6,  ///< The update failed because something about the node is not supported
  topsGraphExecUpdateErrorUnsupportedFunctionChange = 0x7
} topsGraphExecUpdateResult;

typedef enum topsStreamCaptureMode {
  topsStreamCaptureModeGlobal = 0,
  topsStreamCaptureModeThreadLocal,
  topsStreamCaptureModeRelaxed
} topsStreamCaptureMode;
typedef enum topsStreamCaptureStatus {
  topsStreamCaptureStatusNone = 0,    ///< Stream is not capturing
  topsStreamCaptureStatusActive,      ///< Stream is actively capturing
  topsStreamCaptureStatusInvalidated  ///< Stream is part of a capture sequence that has been
                                     ///< invalidated, but not terminated
} topsStreamCaptureStatus;

typedef enum topsStreamUpdateCaptureDependenciesFlags {
  topsStreamAddCaptureDependencies = 0,  ///< Add new nodes to the dependency set
  topsStreamSetCaptureDependencies,      ///< Replace the dependency set with the new nodes
} topsStreamUpdateCaptureDependenciesFlags;

// doxygen end graph API
/**
 * @}
 */


// @cond INTERNAL
/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 * @defgroup StreamO Stream Ordered Memory Allocator
 * @{
 * @ingroup Memory
 * This section describes Stream Ordered Memory Allocator functions of TOPS runtime API.
 *
 * The asynchronous allocator allows the user to allocate and free in stream order.
 * All asynchronous accesses of the allocation must happen between the stream executions of
 * the allocation and the free. If the memory is accessed outside of the promised stream order,
 * a use before allocation / use after free error  will cause undefined behavior.
 *
 * The allocator is free to reallocate the memory as long as it can guarantee that compliant memory
 * accesses will not overlap temporally. The allocator may refer to internal stream ordering as well
 * as inter-stream dependencies (such as TOPS events and null stream dependencies) when establishing
 * the temporal guarantee. The allocator may also insert inter-stream dependencies to establish
 * the temporal guarantee.  Whether or not a device supports the integrated stream ordered memory
 * allocator may be queried by calling @p topsDeviceGetAttribute with the device attribute
 * @p topsDeviceAttributeMemoryPoolsSupported
 *
 * @note  APIs in this section are implemented on Linux, under development on Windows.
 */

/**
 * @brief Allocates memory with stream ordered semantics
 *
 * Inserts a memory allocation operation into @p stream.
 * A pointer to the allocated memory is returned immediately in *dptr.
 * The allocation must not be accessed until the the allocation operation completes.
 * The allocation comes from the memory pool associated with the stream's device.
 *
 * @note The default memory pool of a device contains device memory from that device.
 * @note Basic stream ordering allows future work submitted into the same stream to use the
 *  allocation. Stream query, stream synchronize, and TOPS events can be used to guarantee that
 *  the allocation operation completes before work submitted in a separate stream runs.
 * @note During stream capture, this function results in the creation of an allocation node.
 *  In this case, the allocation is owned by the graph instead of the memory pool. The memory
 *  pool's properties are used to set the node's creation parameters.
 *
 * @param [out] dev_ptr  Returned device pointer of memory allocation
 * @param [in] size      Number of bytes to allocate
 * @param [in] stream    The stream establishing the stream ordering contract and
 *                       the memory pool to allocate from
 * @param [in] flags     Flags to control allocation.
 *                       [63:56] Memory bank for affinity.
 *                       [55:0]  flags to control allocation.
 *
 * @return #topsSuccess, #topsErrorInvalidValue, #topsErrorNotSupported, #topsErrorOutOfMemory
 *
 * @see topsMallocFromPoolAsync, topsFreeAsync, topsMemPoolTrimTo, topsMemPoolGetAttribute,
 * topsDeviceSetMemPool, topsMemPoolSetAttribute, topsMemPoolSetAccess, topsMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMallocAsync(void** dev_ptr, size_t size, topsStream_t stream, uint64_t flags);
/**
 * @brief Frees memory with stream ordered semantics
 *
 * Inserts a free operation into @p stream.
 * The allocation must not be used after stream execution reaches the free.
 * After this API returns, accessing the memory from any subsequent work launched on the GPU
 * or querying its pointer attributes results in undefined behavior.
 *
 * @note During stream capture, this function results in the creation of a free node and
 * must therefore be passed the address of a graph allocation.
 *
 * @param [in] dev_ptr Pointer to device memory to free
 * @param [in] stream  The stream, where the destruction will occur according to the execution order
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorNotSupported
 *
 * @see topsMallocFromPoolAsync, topsMallocAsync, topsMemPoolTrimTo, topsMemPoolGetAttribute,
 * topsDeviceSetMemPool, topsMemPoolSetAttribute, topsMemPoolSetAccess, topsMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsFreeAsync(void* dev_ptr, topsStream_t stream);
/**
 * @brief Releases freed memory back to the OS
 *
 * Releases memory back to the OS until the pool contains fewer than @p min_bytes_to_keep
 * reserved bytes, or there is no more memory that the allocator can safely release.
 * The allocator cannot release OS allocations that back outstanding asynchronous allocations.
 * The OS allocations may happen at different granularity from the user allocations.
 *
 * @note: Allocations that have not been freed count as outstanding.
 * @note: Allocations that have been asynchronously freed but whose completion has
 * not been observed on the host (eg. by a synchronize) can count as outstanding.
 *
 * @param[in] mem_pool          The memory pool to trim allocations
 * @param[in] min_bytes_to_hold If the pool has less than min_bytes_to_hold reserved,
 * then the TrimTo operation is a no-op.  Otherwise the memory pool will contain
 * at least min_bytes_to_hold bytes reserved after the operation.
 *
 * @returns #topsSuccess, #topsErrorInvalidValue
 *
 * @see topsMallocFromPoolAsync, topsMallocAsync, topsFreeAsync, topsMemPoolGetAttribute,
 * topsDeviceSetMemPool, topsMemPoolSetAttribute, topsMemPoolSetAccess, topsMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolTrimTo(topsMemPool_t mem_pool, size_t min_bytes_to_hold);
/**
 * @brief Sets attributes of a memory pool
 *
 * Supported attributes are:
 * - @p topsMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
 *                                  Amount of reserved memory in bytes to hold onto before trying
 *                                  to release memory back to the OS. When more than the release
 *                                  threshold bytes of memory are held by the memory pool, the
 *                                  allocator will try to release memory back to the OS on the
 *                                  next call to stream, event or context synchronize. (default 0)
 * - @p topsMemPoolReuseFollowEventDependencies: (value type = int)
 *                                  Allow @p topsMallocAsync to use memory asynchronously freed
 *                                  in another stream as long as a stream ordering dependency
 *                                  of the allocating stream on the free action exists.
 *                                  TOPS events and null stream interactions can create the required
 *                                  stream ordered dependencies. (default enabled)
 * - @p topsMemPoolReuseAllowOpportunistic: (value type = int)
 *                                  Allow reuse of already completed frees when there is no dependency
 *                                  between the free and allocation. (default enabled)
 * - @p topsMemPoolReuseAllowInternalDependencies: (value type = int)
 *                                  Allow @p topsMallocAsync to insert new stream dependencies
 *                                  in order to establish the stream ordering required to reuse
 *                                  a piece of memory released by @p topsFreeAsync (default enabled).
 *
 * @param [in] mem_pool The memory pool to modify
 * @param [in] attr     The attribute to modify
 * @param [in] value    Pointer to the value to assign
 *
 * @returns #topsSuccess, #topsErrorInvalidValue
 *
 * @see topsMallocFromPoolAsync, topsMallocAsync, topsFreeAsync, topsMemPoolGetAttribute,
 * topsMemPoolTrimTo, topsDeviceSetMemPool, topsMemPoolSetAccess, topsMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolSetAttribute(topsMemPool_t mem_pool, topsMemPoolAttr attr, void* value);
/**
 * @brief Gets attributes of a memory pool
 *
 * Supported attributes are:
 * - @p topsMemPoolAttrReleaseThreshold: (value type = cuuint64_t)
 *                                  Amount of reserved memory in bytes to hold onto before trying
 *                                  to release memory back to the OS. When more than the release
 *                                  threshold bytes of memory are held by the memory pool, the
 *                                  allocator will try to release memory back to the OS on the
 *                                  next call to stream, event or context synchronize. (default 0)
 * - @p topsMemPoolReuseFollowEventDependencies: (value type = int)
 *                                  Allow @p topsMallocAsync to use memory asynchronously freed
 *                                  in another stream as long as a stream ordering dependency
 *                                  of the allocating stream on the free action exists.
 *                                  TOPS events and null stream interactions can create the required
 *                                  stream ordered dependencies. (default enabled)
 * - @p topsMemPoolReuseAllowOpportunistic: (value type = int)
 *                                  Allow reuse of already completed frees when there is no dependency
 *                                  between the free and allocation. (default enabled)
 * - @p topsMemPoolReuseAllowInternalDependencies: (value type = int)
 *                                  Allow @p topsMallocAsync to insert new stream dependencies
 *                                  in order to establish the stream ordering required to reuse
 *                                  a piece of memory released by @p topsFreeAsync (default enabled).
 *
 * @param [in] mem_pool The memory pool to get attributes of
 * @param [in] attr     The attribute to get
 * @param [in] value    Retrieved value
 *
 * @returns  #topsSuccess, #topsErrorInvalidValue
 *
 * @see topsMallocFromPoolAsync, topsMallocAsync, topsFreeAsync,
 * topsMemPoolTrimTo, topsDeviceSetMemPool, topsMemPoolSetAttribute, topsMemPoolSetAccess, topsMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolGetAttribute(topsMemPool_t mem_pool, topsMemPoolAttr attr, void* value);
/**
 * @brief Controls visibility of the specified pool between devices
 *
 * @param [in] mem_pool   Memory pool for access change
 * @param [in] desc_list  Array of access descriptors. Each descriptor instructs the access to enable for a single gpu
 * @param [in] count  Number of descriptors in the map array.
 *
 * @returns  #topsSuccess, #topsErrorInvalidValue
 *
 * @see topsMallocFromPoolAsync, topsMallocAsync, topsFreeAsync, topsMemPoolGetAttribute,
 * topsMemPoolTrimTo, topsDeviceSetMemPool, topsMemPoolSetAttribute, topsMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolSetAccess(topsMemPool_t mem_pool, const topsMemAccessDesc* desc_list, size_t count);
/**
 * @brief Returns the accessibility of a pool from a device
 *
 * Returns the accessibility of the pool's memory from the specified location.
 *
 * @param [out] flags    Accessibility of the memory pool from the specified location/device
 * @param [in] mem_pool   Memory pool being queried
 * @param [in] location  Location/device for memory pool access
 *
 * @returns #topsSuccess, #topsErrorInvalidValue
 *
 * @see topsMallocFromPoolAsync, topsMallocAsync, topsFreeAsync, topsMemPoolGetAttribute,
 * topsMemPoolTrimTo, topsDeviceSetMemPool, topsMemPoolSetAttribute, topsMemPoolSetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolGetAccess(topsMemAccessFlags* flags, topsMemPool_t mem_pool, topsMemLocation* location);
/**
 * @brief Creates a memory pool
 *
 * Creates a TOPS memory pool and returns the handle in @p mem_pool. The @p pool_props determines
 * the properties of the pool such as the backing device and IPC capabilities.
 *
 * By default, the memory pool will be accessible from the device it is allocated on.
 *
 * @param [out] mem_pool    Contains created memory pool
 * @param [in] pool_props   Memory pool properties
 *
 * @note Specifying topsMemHandleTypeNone creates a memory pool that will not support IPC.
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorNotSupported
 *
 * @see topsMallocFromPoolAsync, topsMallocAsync, topsFreeAsync, topsMemPoolGetAttribute, topsMemPoolDestroy,
 * topsMemPoolTrimTo, topsDeviceSetMemPool, topsMemPoolSetAttribute, topsMemPoolSetAccess, topsMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolCreate(topsMemPool_t* mem_pool, const topsMemPoolProps* pool_props);
topsError_t topsDeviceGetMemPool(topsMemPool_t* mem_pool, int device);
/**
 * @brief Destroys the specified memory pool
 *
 * If any pointers obtained from this pool haven't been freed or
 * the pool has free operations that haven't completed
 * when @p topsMemPoolDestroy is invoked, the function will return immediately and the
 * resources associated with the pool will be released automatically
 * once there are no more outstanding allocations.
 *
 * Destroying the current mempool of a device sets the default mempool of
 * that device as the current mempool for that device.
 *
 * @param [in] mem_pool Memory pool for destruction
 *
 * @note A device's default memory pool cannot be destroyed.
 *
 * @returns #topsSuccess, #topsErrorInvalidValue
 *
 * @see topsMallocFromPoolAsync, topsMallocAsync, topsFreeAsync, topsMemPoolGetAttribute, topsMemPoolCreate
 * topsMemPoolTrimTo, topsDeviceSetMemPool, topsMemPoolSetAttribute, topsMemPoolSetAccess, topsMemPoolGetAccess
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolDestroy(topsMemPool_t mem_pool);
/**
 * @brief Allocates memory from a specified pool with stream ordered semantics.
 *
 * Inserts an allocation operation into @p stream.
 * A pointer to the allocated memory is returned immediately in @p dev_ptr.
 * The allocation must not be accessed until the the allocation operation completes.
 * The allocation comes from the specified memory pool.
 *
 * @note The specified memory pool may be from a device different than that of the specified @p stream.
 *
 * Basic stream ordering allows future work submitted into the same stream to use the allocation.
 * Stream query, stream synchronize, and TOPS events can be used to guarantee that the allocation
 * operation completes before work submitted in a separate stream runs.
 *
 * @note During stream capture, this function results in the creation of an allocation node. In this case,
 * the allocation is owned by the graph instead of the memory pool. The memory pool's properties
 * are used to set the node's creation parameters.
 *
 * @param [out] dev_ptr Returned device pointer
 * @param [in] size     Number of bytes to allocate
 * @param [in] mem_pool The pool to allocate from
 * @param [in] stream   The stream establishing the stream ordering semantic
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorNotSupported, #topsErrorOutOfMemory
 *
 * @see topsMallocAsync, topsFreeAsync, topsMemPoolGetAttribute, topsMemPoolCreate
 * topsMemPoolTrimTo, topsDeviceSetMemPool, topsMemPoolSetAttribute, topsMemPoolSetAccess, topsMemPoolGetAccess,
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMallocFromPoolAsync(void** dev_ptr, size_t size, topsMemPool_t mem_pool, topsStream_t stream);
/**
 * @brief Exports a memory pool to the requested handle type.
 *
 * Given an IPC capable mempool, create an OS handle to share the pool with another process.
 * A recipient process can convert the shareable handle into a mempool with @p topsMemPoolImportFromShareableHandle.
 * Individual pointers can then be shared with the @p topsMemPoolExportPointer and @p topsMemPoolImportPointer APIs.
 * The implementation of what the shareable handle is and how it can be transferred is defined by the requested
 * handle type.
 *
 * @note: To create an IPC capable mempool, create a mempool with a @p topsMemAllocationHandleType other
 * than @p topsMemHandleTypeNone.
 *
 * @param [out] shared_handle Pointer to the location in which to store the requested handle
 * @param [in] mem_pool       Pool to export
 * @param [in] handle_type    The type of handle to create
 * @param [in] flags          Must be 0
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorOutOfMemory
 *
 * @see topsMemPoolImportFromShareableHandle
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolExportToShareableHandle(
    void*                      shared_handle,
    topsMemPool_t               mem_pool,
    topsMemAllocationHandleType handle_type,
    unsigned int               flags);
/**
 * @brief Imports a memory pool from a shared handle.
 *
 * Specific allocations can be imported from the imported pool with @p topsMemPoolImportPointer.
 *
 * @note Imported memory pools do not support creating new allocations.
 * As such imported memory pools may not be used in @p topsDeviceSetMemPool
 * or @p topsMallocFromPoolAsync calls.
 *
 * @param [out] mem_pool     Returned memory pool
 * @param [in] shared_handle OS handle of the pool to open
 * @param [in] handle_type   The type of handle being imported
 * @param [in] flags         Must be 0
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorOutOfMemory
 *
 * @see topsMemPoolExportToShareableHandle
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolImportFromShareableHandle(
    topsMemPool_t*              mem_pool,
    void*                      shared_handle,
    topsMemAllocationHandleType handle_type,
    unsigned int               flags);
/**
 * @brief Export data to share a memory pool allocation between processes.
 *
 * Constructs @p export_data for sharing a specific allocation from an already shared memory pool.
 * The recipient process can import the allocation with the @p topsMemPoolImportPointer api.
 * The data is not a handle and may be shared through any IPC mechanism.
 *
 * @param[out] export_data  Returned export data
 * @param[in] dev_ptr       Pointer to memory being exported
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorOutOfMemory
 *
 * @see topsMemPoolImportPointer
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolExportPointer(topsMemPoolPtrExportData* export_data, void* dev_ptr);
/**
 * @brief Import a memory pool allocation from another process.
 *
 * Returns in @p dev_ptr a pointer to the imported memory.
 * The imported memory must not be accessed before the allocation operation completes
 * in the exporting process. The imported memory must be freed from all importing processes before
 * being freed in the exporting process. The pointer may be freed with @p topsFree
 * or @p topsFreeAsync. If @p topsFreeAsync is used, the free must be completed
 * on the importing process before the free operation on the exporting process.
 *
 * @note The @p topsFreeAsync api may be used in the exporting process before
 * the @p topsFreeAsync operation completes in its stream as long as the
 * @p topsFreeAsync in the exporting process specifies a stream with
 * a stream dependency on the importing process's @p topsFreeAsync.
 *
 * @param [out] dev_ptr     Pointer to imported memory
 * @param [in] mem_pool     Memory pool from which to import a pointer
 * @param [in] export_data  Data specifying the memory to import
 *
 * @returns #topsSuccess, #topsErrorInvalidValue, #topsErrorNotInitialized, #topsErrorOutOfMemory
 *
 * @see topsMemPoolExportPointer
 *
 * @warning : This API is marked as beta, meaning, while this is feature complete,
 * it is still open to changes and may have outstanding issues.
 *
 * @note  This API is implemented on Linux, under development on Windows.
 */
topsError_t topsMemPoolImportPointer(
    void**                   dev_ptr,
    topsMemPool_t             mem_pool,
    topsMemPoolPtrExportData* export_data);

// Doxygen end of ordered memory allocator
/**
 * @}
 */
// @endcond

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Runtime Runtime Compilation
 *  @{
 *  This section describes the runtime compilation functions of TOPS runtime API.
 *
 */

// doxygen end Runtime
/**
 * @}
 */

/**
 *-------------------------------------------------------------------------------------------------
 *-------------------------------------------------------------------------------------------------
 *  @defgroup Extension TOPS API
 *  @{
 *  This section describes the runtime Ext functions of TOPS runtime API.
 *
 */

// doxygen end Extension
/**
 * @}
 */

#ifdef __cplusplus
} /* extern "c" */
#endif

#ifdef __GNUC__
#pragma GCC visibility pop
#endif
// doxygen end TOPS API
/**
 *   @}
 */

#else
#error("Must define __TOPS_PLATFORM_ENFLAME__");
#endif

/**
 * @brief: C++ wrapper for topsMalloc
 *
 * Perform automatic type conversion to eliminate need for excessive typecasting (i.e. void**)
 *
 * __TOPS_DISABLE_CPP_FUNCTIONS__ macro can be defined to suppress these
 * wrappers. It is useful for applications which need to obtain decltypes of
 * TOPS runtime APIs.
 *
 * @see topsMalloc
 */
#if defined(__cplusplus) && !defined(__TOPS_DISABLE_CPP_FUNCTIONS__)
template <class T>
static inline topsError_t topsMalloc(T** devPtr, size_t size) {
    return topsMalloc((void**)devPtr, size);
}

template <class T>
topsError_t topsMallocAsync(T** dev_ptr, size_t size, topsStream_t stream, uint64_t flags) {
    return topsMallocAsync((void**)dev_ptr, size, stream, flags);
}

// Provide an override to automatically typecast the pointer type from void**, and also provide a
// default for the flags.
template <class T>
static inline topsError_t topsHostMalloc(T** ptr, size_t size,
                                       unsigned int flags = topsHostMallocDefault) {
    return topsHostMalloc((void**)ptr, size, flags);
}
#endif
#endif

#if USE_PROF_API
#include <tops/tops_prof_str.h>
#endif

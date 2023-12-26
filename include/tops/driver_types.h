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

#ifndef TOPS_INCLUDE_TOPS_DRIVER_TYPES_H
#define TOPS_INCLUDE_TOPS_DRIVER_TYPES_H

#include <tops/tops_common.h>

#if !defined(__TOPSCC_RTC__)
#ifndef __cplusplus
#include <stdbool.h>
#endif
#endif  // !defined(__TOPSCC_RTC__)

typedef void* topsDeviceptr_t;
typedef enum topsChannelFormatKind {
    topsChannelFormatKindSigned = 0,
    topsChannelFormatKindUnsigned = 1,
    topsChannelFormatKindFloat = 2,
    topsChannelFormatKindNone = 3
}topsChannelFormatKind;
typedef struct topsChannelFormatDesc {
    int x;
    int y;
    int z;
    int w;
    enum topsChannelFormatKind f;
}topsChannelFormatDesc;
#define TOPS_TRSA_OVERRIDE_FORMAT 0x01
#define TOPS_TRSF_READ_AS_INTEGER 0x01
#define TOPS_TRSF_NORMALIZED_COORDINATES 0x02
#define TOPS_TRSF_SRGB 0x10
typedef enum topsArray_Format {
    TOPS_AD_FORMAT_UNSIGNED_INT8 = 0x01,
    TOPS_AD_FORMAT_UNSIGNED_INT16 = 0x02,
    TOPS_AD_FORMAT_UNSIGNED_INT32 = 0x03,
    TOPS_AD_FORMAT_SIGNED_INT8 = 0x08,
    TOPS_AD_FORMAT_SIGNED_INT16 = 0x09,
    TOPS_AD_FORMAT_SIGNED_INT32 = 0x0a,
    TOPS_AD_FORMAT_HALF = 0x10,
    TOPS_AD_FORMAT_FLOAT = 0x20
}topsArray_Format;
typedef struct TOPS_ARRAY_DESCRIPTOR {
  size_t Width;
  size_t Height;
  enum topsArray_Format Format;
  unsigned int NumChannels;
}TOPS_ARRAY_DESCRIPTOR;
typedef struct TOPS_ARRAY3D_DESCRIPTOR {
  size_t Width;
  size_t Height;
  size_t Depth;
  enum topsArray_Format Format;
  unsigned int NumChannels;
  unsigned int Flags;
}TOPS_ARRAY3D_DESCRIPTOR;
typedef struct topsArray {
    void* data;  // FIXME: generalize this
    struct topsChannelFormatDesc desc;
    unsigned int type;
    unsigned int width;
    unsigned int height;
    unsigned int depth;
    enum topsArray_Format Format;
    unsigned int NumChannels;
    bool isDrv;
    unsigned int textureType;
}topsArray;
#if !defined(__TOPSCC_RTC__)
typedef struct tops_Memcpy2D {
    size_t srcXInBytes;
    size_t srcY;
    topsMemoryType srcMemoryType;
    const void* srcHost;
    topsDeviceptr_t srcDevice;
    topsArray* srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    topsMemoryType dstMemoryType;
    void* dstHost;
    topsDeviceptr_t dstDevice;
    topsArray* dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
} tops_Memcpy2D;
#endif // !defined(__TOPSCC_RTC__)
typedef struct topsArray* topsArray_t;
typedef topsArray_t topsarray;
typedef const struct topsArray* topsArray_const_t;
typedef struct topsMipmappedArray {
  void* data;
  struct topsChannelFormatDesc desc;
  unsigned int type;
  unsigned int width;
  unsigned int height;
  unsigned int depth;
  unsigned int min_mipmap_level;
  unsigned int max_mipmap_level;
  unsigned int flags;
  enum topsArray_Format format;
} topsMipmappedArray;
typedef struct topsMipmappedArray* topsMipmappedArray_t;
typedef const struct topsMipmappedArray* topsMipmappedArray_const_t;
/**
 * tops resource types
 */
typedef enum topsResourceType {
    topsResourceTypeArray = 0x00,
    topsResourceTypeMipmappedArray = 0x01,
    topsResourceTypeLinear = 0x02,
    topsResourceTypePitch2D = 0x03
}topsResourceType;
typedef enum TOPSresourcetype_enum {
    TOPS_RESOURCE_TYPE_ARRAY           = 0x00, /**< Array resoure */
    TOPS_RESOURCE_TYPE_MIPMAPPED_ARRAY = 0x01, /**< Mipmapped array resource */
    TOPS_RESOURCE_TYPE_LINEAR          = 0x02, /**< Linear resource */
    TOPS_RESOURCE_TYPE_PITCH2D         = 0x03  /**< Pitch 2D resource */
} TOPSresourcetype;
/**
 * tops address modes
 */
typedef enum TOPSaddress_mode_enum {
    TOPS_TR_ADDRESS_MODE_WRAP   = 0,
    TOPS_TR_ADDRESS_MODE_CLAMP  = 1,
    TOPS_TR_ADDRESS_MODE_MIRROR = 2,
    TOPS_TR_ADDRESS_MODE_BORDER = 3
} TOPSaddress_mode;
/**
 * tops filter modes
 */
typedef enum TOPSfilter_mode_enum {
    TOPS_TR_FILTER_MODE_POINT  = 0,
    TOPS_TR_FILTER_MODE_LINEAR = 1
} TOPSfilter_mode;
/**
 * Memory copy types
 *
 */
#if !defined(__TOPSCC_RTC__)
typedef enum topsMemcpyKind {
    topsMemcpyHostToHost = 0,
    topsMemcpyHostToDevice = 1,
    topsMemcpyDeviceToHost = 2,
    topsMemcpyDeviceToDevice = 3,
    topsMemcpyDefault = 4
} topsMemcpyKind;
typedef enum topsFunction_attribute {
    TOPS_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
    TOPS_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
    TOPS_FUNC_ATTRIBUTE_CONST_SIZE_BYTES,
    TOPS_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES,
    TOPS_FUNC_ATTRIBUTE_NUM_REGS,
    TOPS_FUNC_ATTRIBUTE_PTX_VERSION,
    TOPS_FUNC_ATTRIBUTE_BINARY_VERSION,
    TOPS_FUNC_ATTRIBUTE_CACHE_MODE_CA,
    TOPS_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    TOPS_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
    TOPS_FUNC_ATTRIBUTE_MAX
} topsFunction_attribute;

typedef enum topsPointer_attribute {
    TOPS_POINTER_ATTRIBUTE_CONTEXT = 1,   ///< The context on which a pointer was allocated
                                         ///< @warning - not supported in TOPS
    TOPS_POINTER_ATTRIBUTE_MEMORY_TYPE,   ///< memory type describing location of a pointer
    TOPS_POINTER_ATTRIBUTE_DEVICE_POINTER,///< address at which the pointer is allocated on device
    TOPS_POINTER_ATTRIBUTE_HOST_POINTER,  ///< address at which the pointer is allocated on host
    TOPS_POINTER_ATTRIBUTE_P2P_TOKENS,    ///< A pair of tokens for use with linux kernel interface
                                         ///< @warning - not supported in TOPS
    TOPS_POINTER_ATTRIBUTE_SYNC_MEMOPS,   ///< Synchronize every synchronous memory operation
                                         ///< initiated on this region
    TOPS_POINTER_ATTRIBUTE_BUFFER_ID,     ///< Unique ID for an allocated memory region
    TOPS_POINTER_ATTRIBUTE_IS_MANAGED,    ///< Indicates if the pointer points to managed memory
    TOPS_POINTER_ATTRIBUTE_DEVICE_ORDINAL,///< device ordinal of a device on which a pointer
                                         ///< was allocated or registered
    TOPS_POINTER_ATTRIBUTE_IS_LEGACY_TOPS_IPC_CAPABLE, ///< if this pointer maps to an allocation
                                                     ///< that is suitable for topsIpcGetMemHandle
                                                     ///< @warning - not supported in TOPS
    TOPS_POINTER_ATTRIBUTE_RANGE_START_ADDR,///< Starting address for this requested pointer
    TOPS_POINTER_ATTRIBUTE_RANGE_SIZE,      ///< Size of the address range for this requested pointer
    TOPS_POINTER_ATTRIBUTE_MAPPED,          ///< tells if this pointer is in a valid address range
                                           ///< that is mapped to a backing allocation
    TOPS_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES,///< Bitmask of allowed topsmemAllocationHandleType
                                           ///< for this allocation @warning - not supported in TOPS
    TOPS_POINTER_ATTRIBUTE_IS_GCU_DIRECT_RDMA_CAPABLE, ///< returns if the memory referenced by
                                           ///< this pointer can be used with the GCUDirect RDMA API
                                           ///< @warning - not supported in TOPS
    TOPS_POINTER_ATTRIBUTE_ACCESS_FLAGS,    ///< Returns the access flags the device associated with
                                           ///< for the corresponding memory referenced by the ptr
    TOPS_POINTER_ATTRIBUTE_MEMPOOL_HANDLE,  ///< Returns the mempool handle for the allocation if
                                            ///< it was allocated from a mempool
                                            ///< @warning - not supported in TOPS
    TOPS_POINTER_ATTRIBUTE_MEM_BANK  ///< Returns the memory bank for the device
                                     ///< memory - use int data type
} topsPointer_attribute;

#endif  // !defined(__TOPSCC_RTC__)

#endif

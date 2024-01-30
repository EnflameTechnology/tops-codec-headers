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

#ifndef TOPS_INCLUDE_TOPS_TOPS_EXT_H
#define TOPS_INCLUDE_TOPS_TOPS_EXT_H
#include <tops/tops_runtime_api.h>
#include <tops/tops_ext_experimental.h>
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

typedef struct itopsExecutable_t *topsExecutable_t;
typedef struct itopsResource_t *topsResource_t;

enum topsResLocMode {
  TOPS_RES_LOC__SAME = 0, /* alloc all items on same engine */
  TOPS_RES_LOC__DIST,     /* alloc all items on each engine */
};

/* bit mask */
enum topsHbmPolicy {
  TOPS_HBM_POLICY__BEST_FIT = 0x0, /* best to alloc HBM from MC affinitive with
                                      CDMA, try other MC on failure */
  TOPS_HBM_POLICY__AFFINITY_ONLY =
      0x1,                     /* only try to alloc HBM on affinity CDMA */
  TOPS_HBM_POLICY__NONE = 0x2, /* try all HBM from MC0 to last */
};

enum topsResBundlMode {
  TOPS_RES_BUNDLE_MODE__UNKNOWN = 0, /* original non-VG mode */
  TOPS_RES_BUNDLE_MODE__DEVICE,  /* exclusive full raw device based resources */
  TOPS_RES_BUNDLE_MODE__CLUSTER, /* exclusive full raw cluster based resources
                                  */
  TOPS_RES_BUNDLE_MODE__VG,   /* exclusive full best-fit PG based resources */
  TOPS_RES_BUNDLE_MODE__PG,   /* exclusive user specified PG resources */
  TOPS_RES_BUNDLE_MODE__USER, /* full user defined resources */
};

/* use to config topsResourceRequest->res_bundle_cfg */
/* All HW resources required by a task in runtime (instance of executable)
 * Each task has one virtual device
 */
typedef struct topsResourceBundle {
  enum topsResBundlMode mode;    /* resource bundle mode */
  uint64_t sys_mem_size;         /* Host mem size per resource bundle */
  uint64_t hbm_mem_size;         /* HBM size per resource bundle */
  enum topsHbmPolicy hbm_policy; /* HBM affinity or not */
  uint32_t hcve_num;             /* HCVG-HCVE num per resource bundle */
  enum topsResLocMode hcve_loc;  /* HCVE mode, 0=alloc on same HCVG engine,
                                      1=distributed on each engine */
  uint32_t vdec_core_num;        /* VDEC-core num per resource bundle */
  union {
    struct {
      uint32_t cluster_num; /* Cluster count needed for this resource bundle */
    } cluster;              /* TOPS_RES_BUNDLE_MODE__CLUSTER */
    struct {
      uint32_t sip_num;  /* SIP count needed for this resource bundle */
      uint32_t cdma_num; /* CDMA engine count needed for this resource bundle */
      uint32_t cdma_vc_num; /* CDMA VC count per CDMA engine needed for this
                            resource bundle */
    } vg;                   /* TOPS_RES_BUNDLE_MODE__VG */
    struct {
      uint32_t pg_mask; /* Cluster count needed for this resource bundle */
    } pg;               /* TOPS_RES_BUNDLE_MODE__PG */
  } u;
} topsResourceBundle_t;

/**
 * topsResourceRequest descriptor
 *
 */
typedef struct topsResourceRequest {
  uint64_t cluster_count;
  uint8_t compute_res_claim;
  bool need_alloc_cluster;
  bool mem_alloc_affinity_only;
  uint8_t res_bundle_cfg[512];
} topsResourceRequest_t;

/**
 * topsExecutableInfoType types
 *
 */
typedef enum topsExecutableInfoType {
  topsExecutableInfoInputCount = 0x0,
  topsExecutableInfoOutputCount,
  topsExecutableInfoInputSizeList,
  topsExecutableInfoOutputSizeList,
  topsExecutableInfoInputRank,
  topsExecutableInfoInputDimsList,
  topsExecutableInfoOutputRank,
  topsExecutableInfoOutputDimsList,
  topsExecutableInfoInputDataTypeList,
  topsExecutableInfoOutputDataTypeList,
  topsExecutableInfoTensorTableSize,
  topsExecutableInfoInputMinDimsList,
  topsExecutableInfoInputMaxDimsList,
  topsExecutableInfoOutputMaxDimsList,
  topsExecutableInfoBlockCount,
  topsExecutableInfoInputBankList,
  topsExecutableInfoOutputBankList,
  topsExecutableInfoThreadNumberPerBlock,
} topsExecutableInfoType_t;

/**
 * topsResourceBundleInfoType types
 *
 */
typedef enum topsResourceBundleInfoType {
  topsResourceBundleProcessorCount = 0x0,
} topsResourceBundleInfoType_t;

#define MAX_CDMA_ENGINE_NUM_ON_20 4

typedef struct topsKernelDescriptor {
  unsigned int kernarg_size;
  unsigned int shared_mem_size;
  unsigned int cdma_vc_num[MAX_CDMA_ENGINE_NUM_ON_20];
  unsigned int sdma_vc_num;
  unsigned int sip_mbx_num;
  unsigned char factor;           // bool
  unsigned int sip_code_offset;   // launch_entry, intba
  unsigned int exception_offset;  // 8

  unsigned int sip_mode;
  unsigned int stack_size;
  // tops runtime fields of param layout: args_num = readonly_num + writable_num
  unsigned int kernarg_readonly_num;
  unsigned int kernarg_writable_num;

  unsigned int global_gsync_entry_count;
  unsigned int global_gsync_counter_count;
  unsigned int shared_gsync_entry_count;
  unsigned int shared_gsync_counter_count;

  unsigned int shared_queue_count;
  unsigned int shared_latch_count;
  unsigned int shared_barrier_count;
  unsigned int shared_mutex_count;

  unsigned int global_queue_count;
  unsigned int global_latch_count;
  unsigned int global_barrier_count;
  unsigned int global_mutex_count;

} topsKernelDescriptor_t;

/**
 * topsExtLaunchParams descriptor
 *
 */
typedef struct topsExtLaunchParams {
  int cluster_id;
  void *func;  // l3 addr
  size_t code_size;
  void *args;  // l4 addr
  size_t args_size;
  void *rodata;  // l4 addr
  size_t rodata_size;
  void *tensor_table; // l4 addr
  size_t tensor_table_size;
  size_t debug_id;
  topsKernelDescriptor_t descriptor;
  size_t update_count;
  uint32_t *update_offset;
} topsExtLaunchParams_t;

/**
 * Reduce Operation define.
 */
enum topsReduceOpType {
  REDUCE_SUM = 0,
  REDUCE_PROD,
  REDUCE_MAX,
  REDUCE_MIN,
  REDUCE_UNKNOW = 4,
};

/**
 * Reduce Data type define.
 */
enum topsReduceDataType {
  INT8 = 0,
  CHAR = 0,
  UINT8 = 1,
  INT32 = 2,
  INT = 2,
  UINT32 = 3,
  INT64 = 4,
  UINT64 = 5,
  FLOAT16 = 6,
  HALF = 6,
  FLOAT32 = 7,
  FLOAT = 7,
  LFLOAT64 = 8,
  DOUBLE = 8,
  BFLOAT16 = 9,
  UNKNOW = 10,
};

/**
 * topsExecutable section information
 */
typedef struct topsExtExecutableSectionInfo {
  int64_t index;
  uint64_t sh_type; // section header type
  uint64_t size; // section size
  uint64_t device_load_size; // the size of data load to device
  uint64_t offset_in_exe; // offset of the section in executable
  uint64_t loaded_address; // address of the section when load
  uint64_t key; // 0 if no key
  bool  is_sub_section; // sub section flag
} topsExtExecutableSectionInfo_t;

/**
 * topsExecutable section header types
 */
typedef enum topsExtExecutableSectionHeaderType {
  SHT_NULL_ = 0,
  SHT_HEADER,
  SHT_RESOURCE,
  SHT_INPUT,
  SHT_OUT,
  SHT_CONSTANT,
  SHT_SIPCODE,
  SHT_SIP_FUNC_PARAM,
  SHT_SIP_FUNC_PARAM_UPDATE,
  SHT_STREAM,
  SHT_PACKET, // sub section of SHT_CLUSTER
  SHT_PACKET_UPDATE, // sub section of SHT_CLUSTER
  SHT_CLUSTER,
  SHT_JITBINARY,
  SHT_CPU_FUNC_PARAM,
  SHT_CPU_FUNC_DATA,
  SHT_CPU_FUNCTION,
  SHT_PROFILE,
  SHT_HOST_CONST,
  SHT_TASK, // sub section of SHT_CLUSTER
  SHT_SIP_CALLBACK,
  SHT_TARGET_RESOURCE,
  SHT_TENSOR_TABLE,
  SHT_KERNEL_ASSERT_INFO,
  SHT_RAND_STATE,
  SHT_HOST_PROGRAM,
  SHT_RODATA,
  SHT_MANAGED_CONSTANT,
  SHT_USER5,
  SHT_USER6,
  SHT_USER7,
  SHT_USER8,
  SHT_USER9,
  SHT_LAST_TYPE, // limit the range, not exist actually
} topsExtExecutableSectionHeaderType_t;

/**
 * max dims of input/output
 */
#define TOPS_SHAPE_INFER_MAX_DIMS 8

/**
 * shape of input/output
 */
typedef struct topsShape {
  int rank;
  int dims[TOPS_SHAPE_INFER_MAX_DIMS];
} topsShape_t;

// Doxygen end group GlobalDefsExt
/**  @} */

/**
 *  @addtogroup Extension TOPS API
 *  @{
 *  @ingroup Extension
 */

/**
 * @brief Set dimension for device memory.
 *
 * Limitation: devPtr only support allocated by topsMalloc*
 *
 * @param [in] devPtr The device memory to be set.
 * @param [in] dims The dimension to be set.
 * @param [in] dims_count The dimension size to be set.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsMemorySetDims(const void *devPtr, int64_t *dims,
                              size_t dims_count);

/**
 * @brief Get dimension of device memory.
 *
 * Limitation: devPtr only support allocated by topsMalloc*
 *
 * @param [in]  devPtr The device memory to be set.
 * @param [out] dims The dimension pointer list to be get.
 * @param [out] dims_count The dimension rank pointer list to be get.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsMemoryGetDims(const void *devPtr, int64_t *dims,
                              size_t *dims_count);

/**
 * @brief Create an executable with specified binary data and size.
 *
 * @param [in]  bin The pointer to the binary data.
 * @param [in]  size The size of the binary data.
 * @param [out] exe Pointer to get new executable.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: Ownership of the pointer to new executable is transferred to the
 * caller. Caller need call topsDestroyExecutable to destroy this pointer when
 * no longer use it.
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
TOPS_PUBLIC_API
topsError_t topsCreateExecutable(topsExecutable_t *exe, const void *bin,
                                 size_t size);

/**
 * @brief Create an executable with specified file.
 *
 * @param [in]  filepath The name of binary file.
 * @param [out] exe Pointer to get new executable.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: Ownership of the pointer to new executable is transferred to the
 * caller. Caller need call topsDestroyExecutable to destroy this pointer when
 * no longer use it.
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
TOPS_PUBLIC_API
topsError_t topsCreateExecutableFromFile(topsExecutable_t *exe,
                                         const char *filepath);

/**
 * @brief Destroy and clean up an executable.
 *
 * @param [in] exe Pointer to executable to destroy.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsDestroyExecutable(topsExecutable_t exe);

/**
 * @brief Create a resource bundle with specified request.
 * If device is in reset status, this method will retry until reset finish.
 *
 * @param [out] res Pointer to get new resource bundle or nullptr if failed.
 * @param [in]  req Requested resource of allocated resource bundle.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: Ownership of the pointer to new resource bundle is transferred
 * to the caller. Caller need call topsDestroyResource to destroy this pointer
 * when no longer use it.
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
TOPS_PUBLIC_API
topsError_t topsCreateResource(topsResource_t *res, topsResourceRequest_t req);

/**
 * @brief Create a new resource bundle with specified target resource.
 * If device is in reset status, this method will retry until reset finish.
 *
 * @param [out] res Pointer to get new resource bundle or nullptr if failed.
 * @param [in]  exe Pointer to the executable to get target pointer
 *            which contains the requested resource.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: Ownership of the pointer to new resource bundle is transferred
 * to the caller. Caller need call topsDestroyResource
 * to destroy this pointer when no longer use it.
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
TOPS_PUBLIC_API
topsError_t topsCreateResourceForExecutable(topsResource_t *res,
                                            topsExecutable_t exe);

/**
 * @brief Destroy and clean up a resource bundle.
 *
 * @param [in] res Pointer to resource bundle to destroy.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsDestroyResource(topsResource_t res);

/**
 * @brief Query resource bundle attribute.
 *
 * @param [in]  res Pointer to resource bundle to query.
 * @param [in]  type Type to query.
 * @param [out] data Pointer to query output data.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsResourceBundleGetAttribute(topsResource_t res,
                                           topsResourceBundleInfoType_t type,
                                           uint64_t *data);

/**
 *  @brief Allocate memory on the res_bundle affinity memory bank
 *
 *  If size is 0, no memory is allocated, *ptr returns non-nullptr, and topsSuccess
 * is returned.
 *
 *  Use topsFree to release ptr
 *
 *  @param [out] ptr Pointer to the allocated memory
 *  @param [in]  size Requested memory size
 *  @param [in]  res res_bundle
 *
 *  @return #topsSuccess, #topsErrorOutOfMemory, #topsErrorInvalidValue (bad
 * context, null *ptr)
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
TOPS_PUBLIC_API
topsError_t topsMallocForResource(void **ptr, size_t size, topsResource_t res);

/**
 * @brief Asynchronously run a executable.
 *
 * Run an executable with given inputs and outputs for training.
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  res Pointer to res_bundle object.
 * @param [in]  inputs Inputs of executable.
 * @param [in]  input_count Inputs count of executable.
 * @param [in]  input_dims Inputs dims List of executable.
 * @param [in]  input_rank Inputs dims rank of executable.
 * @param [in]  outputs Outputs of executable.
 * @param [in]  output_count Outputs count of executable.
 * @param [in]  stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: default use runExe with Operator mode (not support dynamic shape),
 * and if res is nullptr, use default res_bundle
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
TOPS_PUBLIC_API
topsError_t topsLaunchExecutableV2(topsExecutable_t exe, topsResource_t res,
                                   void **inputs, size_t input_count,
                                   int64_t *input_dims, size_t *input_rank,
                                   void **outputs, size_t output_count,
                                   topsStream_t stream);

/**
 * @brief Asynchronously run a executable.
 *
 * Run an executable with given inputs and outputs for training.
 *
 * Limitation: for performance, output_dims/output_rank should be set to
 * nullptr. if users set output_dims/output_rank as no-zero, topsLaunchExecutableV3
 * will insert a blocking mode hostcallback operation to get output_dims and output_rank
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  res Pointer to res_bundle object.
 * @param [in]  inputs Inputs of executable.
 * @param [in]  input_count Inputs count of executable.
 * @param [in]  input_dims Inputs dims List of executable.
 * @param [in]  input_rank Inputs dims rank of executable.
 * @param [in]  outputs Outputs of executable.
 * @param [in]  output_count Outputs count of executable.
 * @param [out] output_dims Outputs dims List of executable.
 * @param [out] output_rank Outputs dims rank of executable.
 * @param [in]  stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: default use runExe with Graph mode(support dynamic shape),
 * and if res is nullptr, use default res_bundle
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
TOPS_PUBLIC_API
topsError_t topsLaunchExecutableV3(topsExecutable_t exe, topsResource_t res,
                                   void **inputs, size_t input_count,
                                   int64_t *input_dims, size_t *input_rank,
                                   void **outputs, size_t output_count,
                                   int64_t *output_dims, size_t *output_rank,
                                   topsStream_t stream);

/**
 * @brief Asynchronously run a executable.
 *
 * Run an executable with given inputs and outputs for training.
 *
 * Limitation: for performance, output_dims/output_rank should be set to
 * nullptr. if user set output_dims/output_rank no-zero, topsLaunchExecutable
 *             may synchronize to get output_dims and output_rank
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  res Pointer to res_bundle object.
 * @param [in]  inputs Inputs of executable.
 * @param [in]  input_count Inputs count of executable.
 * @param [in]  input_dims Inputs dims List of executable.
 * @param [in]  input_rank Inputs dims rank of executable.
 * @param [in]  outputs Outputs of executable.
 * @param [in]  output_count Outputs count of executable.
 * @param [out] output_dims Outputs dims List of executable.
 * @param [out] output_rank Outputs dims rank of executable.
 * @param [in]  stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: default use runExe with Graph mode(support dynamic shape),
 * and if res is nullptr, use default res_bundle
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
TOPS_PUBLIC_API
topsError_t topsLaunchExecutable(topsExecutable_t exe, topsResource_t res,
                                 void **inputs, size_t input_count,
                                 int64_t *input_dims, size_t *input_rank,
                                 void **outputs, size_t output_count,
                                 int64_t *output_dims, size_t *output_rank,
                                 topsStream_t stream);

/**
 * @brief Asynchronously run a executable.
 *
 * Run an executable with given inputs and outputs for training.
 *
 * Limitation: for performance, output_dims/output_rank should be set to
 * nullptr. if user set output_dims/output_rank no-zero,
 * topsLaunchExecutableWithConstData may synchronize to get output_dims and
 * output_rank
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  res Pointer to res_bundle object.
 * @param [in]  inputs Inputs of executable.
 * @param [in]  input_count Inputs count of executable.
 * @param [in]  input_dims Inputs dims List of executable.
 * @param [in]  input_rank Inputs dims rank of executable.
 * @param [in]  outputs Outputs of executable.
 * @param [in]  output_count Outputs count of executable.
 * @param [out] output_dims Outputs dims List of executable.
 * @param [out] output_rank Outputs dims rank of executable.
 * @param [in]  const_datas Const_datas of executable.
 * @param [in]  const_datas_count Const_datas count of executable.
 * @param [in]  stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: default use runExe with Graph mode(support dynamic shape),
 * and if res is nullptr, use default res_bundle
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
TOPS_PUBLIC_API
topsError_t topsLaunchExecutableWithConstData(
    topsExecutable_t exe, topsResource_t res, void **inputs, size_t input_count,
    int64_t *input_dims, size_t *input_rank, void **outputs,
    size_t output_count, int64_t *output_dims, size_t *output_rank,
    void **const_datas, size_t const_datas_count, topsStream_t stream);

/**
 * @brief Asynchronously run a executable.
 *
 * Run an executable with given inputs and outputs for training.
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  res Pointer to res_bundle object.
 * @param [in]  inputs Inputs of executable.
 * @param [in]  input_count Inputs count of executable.
 * @param [in]  input_dims Inputs dims List of executable.
 * @param [in]  input_rank Inputs dims rank of executable.
 * @param [in]  outputs Outputs of executable.
 * @param [in]  output_count Outputs count of executable.
 * @param [in]  const_datas Const_datas of executable.
 * @param [in]  const_datas_count Const_datas count of executable.
 * @param [in]  stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: default use runExe with Operator mode (not support dynamic shape),
 * and if res is nullptr, use default res_bundle
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
TOPS_PUBLIC_API
topsError_t topsLaunchExecutableWithConstDataV2(
    topsExecutable_t exe, topsResource_t res, void **inputs, size_t input_count,
    int64_t *input_dims, size_t *input_rank, void **outputs,
    size_t output_count, void **const_datas, size_t const_datas_count,
    topsStream_t stream);

/**
 * @brief Asynchronously run a executable.
 *
 * Run an executable with given inputs and outputs for training.
 *
 * Limitation: for performance, output_dims/output_rank should be set to
 * nullptr. if users set output_dims/output_rank as no-zero,
 * topsLaunchExecutableWithConstDataV3 will insert a blocking mode hostcallback
 * operation to get output_dims and output_rank
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  res Pointer to res_bundle object.
 * @param [in]  inputs Inputs of executable.
 * @param [in]  input_count Inputs count of executable.
 * @param [in]  input_dims Inputs dims List of executable.
 * @param [in]  input_rank Inputs dims rank of executable.
 * @param [in]  outputs Outputs of executable.
 * @param [in]  output_count Outputs count of executable.
 * @param [out] output_dims Outputs dims List of executable.
 * @param [out] output_rank Outputs dims rank of executable.
 * @param [in]  const_datas Const_datas of executable.
 * @param [in]  const_datas_count Const_datas count of executable.
 * @param [in]  stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: default use runExe with Graph mode(support dynamic shape),
 * and if res is nullptr, use default res_bundle
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
TOPS_PUBLIC_API
topsError_t topsLaunchExecutableWithConstDataV3(
    topsExecutable_t exe, topsResource_t res, void **inputs, size_t input_count,
    int64_t *input_dims, size_t *input_rank, void **outputs,
    size_t output_count, int64_t *output_dims, size_t *output_rank,
    void **const_datas, size_t const_datas_count, topsStream_t stream);

/**
 * @brief Get Const Managed Data
 *
 * @param [out]  numOptions Pointer to ConstManagedData array size.
 * @param [in]   exe Pointer to executable object.
 * @param [out]  name ConstManaged pair of name.
 * @param [out]  address ConstManaged pair of address.
 * @param [out]  size ConstManaged address pair of size.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: call twice, first get numOptions, Second get the rest
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
TOPS_PUBLIC_API
topsError_t topsExecutableGetConstManagedData(topsExecutable_t exe,
                                              unsigned int *numOptions,
                                              char **name, void **address,
                                              uint64_t *size, int64_t *uid,
                                              void **flag);

/**
 * @brief Update Constant Section Key
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: must called before save executable to file
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
TOPS_PUBLIC_API
topsError_t topsExecutableUpdateConstantKey(topsExecutable_t exe);

/**
 * @brief Update Executable Runtime Resource
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  res Pointer to res_bundle object.
 * @param [in]  stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: call after refit, will update constant partially by h2d
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
TOPS_PUBLIC_API
topsError_t topsExecutableUpdateRuntimeResource(topsExecutable_t exe,
                                                topsResource_t res,
                                                topsStream_t stream);

/**
 * @brief Load constant data.
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  res Pointer to res_bundle object.
 * @param [out]  dev_ptr Dev_mem of executable.
 * @param [out]  dev_ptr_count Dev_mem count of executable.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * NOTE: pointer variable dev_ptr must initialized to nullptr
 * to call this API, call topsFree dev_ptr[index] to free each
 * device memory, then delete dev_ptr when finish
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
// TOPS_PUBLIC_API
// topsError_t topsExecutableLoadConstData(topsExecutable_t exe, topsResource_t res, void** &dev_ptr,
//                                         size_t *dev_ptr_count);

/**
 * @brief Get runtime dynamic output shape
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  inputs_shape Pointer to input shape.
 * @param [out] outputs_shape Pointer to output shape.
 * @param [out] infer_success flag to indicate shape infer result.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: flag indicate shape infer result, true for success and
 * false for failed. for shape infer failed and legacy executable,
 * return static output shape
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.5_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsExecutableGetRuntimeOutputShape(topsExecutable_t exe,
                                                topsShape_t *inputs_shape,
                                                topsShape_t *outputs_shape,
                                                bool *infer_success);

/**
 * @brief get excutable sub function information
 *
 * @param[in] exe Pointer to executable object
 * @param[out] Info_RawData raw data of sub function information
 * @param[out] Info_size  size of Info_RawData
 * @param[out] param_info  count list of input and output for each sub function
 * @param[out] param_count  total count of input and output
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: use only in refit stage 2, user should parse the
 * Info_RawData to get detailed sub function information
 * example: param_info[2, 1, 3, 2, 3, 1] means subfuncA need 2 inputs, 1 output;
 * subfuncB need 3 inputs, 2 outputs; subfuncC need 3 inputs, 1 output
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.5_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsExecutableGetSubFuncInfo(topsExecutable_t exe,
                                         char** info_raw_data,
                                         size_t* info_size,
                                         int** param_info,
                                         int* param_count);

/**
 * @brief get excutable refit flag
 *
 * @param[in] exe Pointer to executable object.
 * @param[out] refit_flag the flag indicate if this executable is refitable
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.5_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsExecutableGetRefitFlag(topsExecutable_t exe, int* refit_flag);

/**
 * @brief call sub function in executable
 *
 * Limitation: inputs and outputs only support alloc by
 *             topsMalloc/topsHostMalloc
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  func_name sub function name
 * @param [in]  inputs Inputs of sub function.
 * @param [in]  input_count Inputs count of sub function.
 * @param [in]  input_dims Inputs dims List of sub function.
 * @param [in]  input_rank Inputs dims rank of sub function.
 * @param [in]  outputs Outputs of sub function.
 * @param [in]  output_count Outputs count of sub function.
 * @param [in]  stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: use only in refit stage 2 for constant preprocessing
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.5_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsExecutableCallSubFunc(topsExecutable_t exe,
                                      const char* func_name, void **inputs,
                                      size_t input_count, int64_t *input_dims,
                                      size_t *input_rank, void **outputs,
                                      size_t output_count, topsStream_t stream);

/**
 * @brief Set default resource_bundle.
 *
 * Limitation: set current thread global resource_bundle,
 *             set resource_bundle before all tops APIs
 *
 * @param [in] res Pointer to res_bundle object.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsDeviceSetResource(topsResource_t res);

/**
 * @brief Get const buffer.
 *
 * Get or init const buffer with special hash_key
 *
 * @param [in]  hash_key value to const buffer special key.
 * @param [in]  init_data Pointer to const raw buffer.
 * @param [in]  size Size to const buffer size.
 * @param [out] ptr Pointer to get const buffer.
 * @param [in]  stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsConstBufferGet(uint64_t hash_key, void *init_data,
                               size_t size, void **ptr, topsStream_t stream);
/**
 * @brief Put const buffer.
 *
 * @param [in] hash_key value to const buffer special key.
 * @param [in] stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsConstBufferPut(uint64_t hash_key, topsStream_t stream);

/**
 * @brief Query executable info.
 *
 * Limitation: user need to allocate memory for data
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  info_type Type to query.
 * @param [out] data Pointer to query output data.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsExecutableQueryInfo(topsExecutable_t exe,
                                    topsExecutableInfoType_t info_type,
                                    uint64_t *data);

/**
 * @brief Query executable info.
 *
 * Limitation: user need to allocate memory for data
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  info_type Type to query.
 * @param [out] data Pointer to query output data.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsExecutableQueryInfoV2(topsExecutable_t exe,
                                      topsExecutableInfoType_t info_type,
                                      int64_t *data);

/**
 * @brief Query executable input name.
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  index Specify which input to query.
 * @param [out] name The name of the input.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsExecutableQueryInputName(topsExecutable_t exe,
                                         int index,
                                         char **name);

/**
 * @brief Query executable output name.
 *
 * @param [in]  exe Pointer to executable object.
 * @param [in]  index Specify which output to query.
 * @param [out] name The name of the output.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsExecutableQueryOutputName(topsExecutable_t exe,
                                          int index,
                                          char **name);

/**
 * @brief Save executable to a specified file.
 *
 * @param [in] exe Pointer to executable object.
 * @param [in] path The name of the file to be used to save executable.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsExecutableSaveToFile(topsExecutable_t exe, const char *path);

/**
 *  @brief Allocate memory on the specify memory bank
 *
 *  If size is 0, no memory is allocated, *ptr returns non-nullptr, and topsSuccess
 * is returned.
 *
 *  NOTE: default bank is 0
 *
 *  Use topsFree to release ptr
 *
 *  @param [out] ptr Pointer to the allocated memory
 *  @param [in]  size Requested memory size
 *  @param [in]  bank memory bank
 *
 *  @return #topsSuccess, #topsErrorOutOfMemory, #topsErrorInvalidValue (bad
 * context, null *ptr)
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
TOPS_PUBLIC_API
topsError_t topsExtMallocWithBank(void **ptr, size_t size, uint64_t bank);

/**
 *  @brief Allocate memory on the specify memory bank
 *
 *  If size is 0, no memory is allocated, *ptr returns non-nullptr, and topsSuccess
 * is returned.
 *
 *  NOTE: default bank is 0
 *
 *  Use topsFree to release ptr
 *
 *  @param [out] ptr Pointer to the allocated memory
 *  @param [in]  size Requested memory size
 *  @param [in]  bank memory bank
 *  @param [in]  flags Type of memory allocation.
 *               flags only support topsDeviceMallocDefault/topsMallocTopDown/topsMallocForbidMergeMove
 *
 *  @return #topsSuccess, #topsErrorOutOfMemory, #topsErrorInvalidValue (bad
 * context, null *ptr)
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
TOPS_PUBLIC_API
topsError_t topsExtMallocWithBankV2(void **ptr, size_t size,
                                    uint64_t bank, unsigned int flags);
/**
 *  @brief Allocate memory on the logical memory bank
 *
 *  If size is 0, no memory is allocated, *ptr returns non-nullptr, and topsSuccess
 * is returned.
 *
 *  NOTE: default bank is 0, the logical bank will be mapped to physical bank
 *
 *  Use topsFree to release ptr
 *
 *  @param [out] ptr Pointer to the allocated memory
 *  @param [in]  size Requested memory size
 *  @param [in]  bank memory bank
 *  @param [in]  flags Type of memory allocation.
 *               flags only support topsDeviceMallocDefault/topsMallocTopDown/topsMallocForbidMergeMove
 *
 *  @return #topsSuccess, #topsErrorOutOfMemory, #topsErrorInvalidValue (bad
 * context, null *ptr)
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
TOPS_PUBLIC_API
topsError_t topsExtMallocWithAffinity(void **ptr, size_t size,
                                      uint64_t bank, unsigned int flags);

/**
 * @brief Asynchronously launch kernel.
 *
 * @param [in] launchParamsList Pointer to launch params List.
 * @param [in] numClusters Size to use cluster count.
 * @param [in] gridDim Grid Dimension to use.
 * @param [in] blockDim Block Dimension to use.
 * @param [in] stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsExtLaunchCooperativeKernelMultiCluster(
    const topsExtLaunchParams_t *launchParamsList, int numClusters,
    dim3 gridDim, dim3 blockDim, topsStream_t stream);

/**
 * @brief Set profile meta data.
 *
 * @param [in] Pointer to profile meta data.
 * @param [in] profile meta data size in bytes.
 * @param [in] compilation id.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsExtSetProfileMeta(uint8_t *meta, uint32_t size,
                                  int64_t compilation_id);

/**
 * @brief get scatter memory config: win_pos and map_ctrl.
 *
 * This interface only works when dev_ptr is created with topsMallocForScatter
 * and init with topsScatterSetSubMem.
 *
 * Limitation: user need to allocate memory win_pos/map_ctrl array
 *
 * @param [in] dev_ptr  Pointer to scatter memory.
 * @param [in] index    Index to sub mem.
 * @param [in] win_pos  Pointer to sub memory window position.
 * @param [in] win_size Pointer to sub memory window position size.
 * @param [in] map_ctrl Pointer to sub memory map ctrl.
 * @param [in] map_size Pointer to sub memory map ctrl size.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsScatterMemoryGetInfo(const void *dev_ptr, int index,
                                     int64_t *win_pos, size_t *win_size,
                                     int64_t *map_ctrl, size_t *map_size);

/**
 * @brief Get the sub memory number in a scatter memory.
 *
 * This interface only works when dev_ptr is created with topsMallocScatter
 * and init with topsScatterSetSubMem.
 *
 * @param [in] dev_ptr Pointer to scatter memory.
 * @param [in] Size    Pointer to sub memory number size.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsScatterMemoryGetSubNum(const void *dev_ptr, size_t *size);

/**
 * Creates scatter memory on local device.
 *
 * Creates device memory object for holding a list of scattered DeviceMemory
 * objects. Memory is not actually allocated. User can call topsScatterPopulateSub
 * to allocate sub device memory and topsScatterGetSubMem to query sub memory objects.
 * Scatter DeviceMemory handles will be automatically processed by runtime API such
 * like topsMemcpy, it can be viewed as a plain memory buffer.
 *
 *  Use topsFree to release ptr
 *
 * A scatter DeviceMemory is invalid until it's fully constructed with correctly
 * splitted sub DeviceMemory objects both in size and dimension (all related
 * DeviceMemory objects have invoked SetDims).
 *
 * @param [in] ptr  Pointer to the allocated memory
 * @param [in] size Requested creation size in bytes.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsMallocScatter(void **ptr, size_t size);

/**
 * clear submemory.
 *
 * This interface will clear all submemory belonged to scatter memory.
 *
 * @param[in] dev_ptr  Pointer to scatter memory.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: This API won't be supported in future.
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
TOPS_PUBLIC_API
topsError_t topsScatterClearSubMemory(const void *dev_ptr);

/**
 * Set a submemory to construct scatter memory.
 *
 * This interface only works when dev_ptr is created with topsMallocScatter API.
 *
 * @param[in] dev_ptr  Pointer to scatter memory.
 * @param[in] sub_dptr The submemory object that user creates. must have invoked
 * SetDims
 * @param[in] win_pos  The anchor of window in the scatter memory holding this
 * submemory.
 * @param[in] win_size The size of window position, the max size is 8.
 * @param[in] map_ctrl The dimension remap of reshaping. It's a natural number
 * up to rank.
 * @param[in] map_size The size of map ctrl, the max size is 8.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: This API won't be supported in future.
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
TOPS_PUBLIC_API
topsError_t topsScatterSetSubMem(const void *dev_ptr, void *sub_dptr,
                                 int64_t *win_pos, size_t win_size,
                                 int64_t *map_ctrl, size_t map_size);

/**
 * Populate scatter memory with sub-memory.
 *
 * This interface only works when dev_ptr is created with topsMallocScatter API.
 * Each invocation populates one sub-memory. The parent scatter memory owns this
 * sub-memory, that all populated sub-memory are freed if parent is free.
 * topsScatterGetSubMem WON'T retain the sub-memory.
 *
 * @param[in] dev_ptr  Pointer to scatter memory.
 * @param[in] bpe memory bpe of single sub-memory.
 * @param[in] bank memory bank of single sub-memory.
 * @param[in] dims The dimension to be set.
 * @param[in] win_pos  The anchor of window in the scatter memory holding this
 * submemory.
 * @param[in] map_ctrl The dimension remap of reshaping. It's a natural number
 * up to rank.
 * @param[in] rank_size The size of map ctrl/dims/win_position, the max size is 8.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsScatterPopulateSub(const void *dev_ptr, size_t bpe,
                                   uint64_t bank, int64_t *dims,
                                   int64_t *win_pos, int64_t *map_ctrl,
                                   size_t rank_size);

/**
 * Inplace scatter memory with sub-memory description.
 *
 * This interface only works when dev_ptr is created with topsMallocScatter API.
 * The parent scatter memory owns this sub-memory, that all populated
 * sub-memory are freed if parent is free.
 *
 * topsScatterGetSubMem WON'T retain the sub-memory.
 *
 * @param[in] dev_ptr  Pointer to scatter memory, if dev_ptr has sub_memory,
 * sub memory must be allocated by topsScatterPopulateSub
 * @param[in] sub_count Number of sub memory
 * @param[in] bpe memory bpe of single sub-memory.
 * @param[in] bank_list memory bank of sub-memory list.
 * @param[in] dims_list The dimension list to be set.
 * @param[in] win_pos_list  The anchor of window list in the scatter memory
 * holding this submemory.
 * @param[in] map_ctrl_list The dimension remap of reshaping list. It's a
 * natural number up to rank.
 * @param[in] rank_size The size of map ctrl/dims/win_position, the max size is
 * 8.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsScatterInplace(const void *dev_ptr, size_t sub_count,
                               size_t bpe, uint64_t *bank_list,
                               int64_t **dims_list, int64_t **win_pos_list,
                               int64_t **map_ctrl_list, size_t rank_size);

/**
 * Get a submemory from scatter memory .
 *
 * This interface only works when DeviceMemory is created with topsMallocScatter
 * API. In fact, user can hold the sub DeviceMemory instead of getting it from
 * scatter.
 *
 * @param[in] dev_ptr  Pointer to scatter memory.
 * @param[in] index    The index returned by ScatterSetSubMemory.
 * @param[in] sub_dptr The submemory object that user set by
 * topsScatterSetSubMem.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsScatterGetSubMem(const void *dev_ptr, int index,
                                 void **sub_dptr);

/**
 * Asynchronously request reduce service.
 *
 * Request reduce service on Device.
 * The lhs, rhs and result should be on the same device
 *
 * @param[in] lhs  Left Hand Side device memory address object.
 * @param[in] rhs  Right Hand Side device memory object.
 * @param[in] result  Result device memory object.
 * @param[in] op  Reduce op type.
 * @param[in] dtype  Reduce data type.
 * @param[in] element_cnt  Count of data to do the calculation
 * @param[in]  stream stream identifier.
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | NO
 *  gcu300 (scorpio)| NO
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
// TOPS_PUBLIC_API
// topsError_t topsMemoryReduceAsync(void *lhs, void *rhs, void *result,
//                                   topsReduceOpType op, topsReduceDataType dtype,
//                                   uint32_t element_cnt, topsStream_t stream);

/**
 *  @brief Prefetch the contents of a device memory to L2 buffer. The API is
 *  available for GCU 3.0 product only, no effect for other products.
 *  A device memory range is defined through device memory pointer and size.
 *  A device memory range can be equal to original device memory or a sub range
 *  of original device memory. Upper-level software needs to guarantee sub memories
 *  have no overlap. Before prefetch, need to make sure the contents on device memory
 *  is ready.
 *
 *  @param[in]  dptr  Global device pointer
 *  @param[in]  size  Global size in bytes
 *
 *  @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryAllocation
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
 *  gcu200 (pavo)   | NO
 *  gcu210 (dorado) | NO
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
TOPS_PUBLIC_API
topsError_t topsMemCachePrefetch(topsDeviceptr_t dptr, size_t size);

/**
 *  @brief Inavlidate L2 buffer cache data. The API is available for GCU 3.0
 *  product only, no effect for other products.
 *  The API deletes L2 buffer cache data which it is prefetched through API
 *  topsMemCachePrefetch(). The original device memory keeps no change.
 *  A device memory range is defined through device memory pointer and size.
 *
 *  @param[out]  dptr  Global device pointer
 *  @param[out]  size  Global size in bytes
 *
 *  @return #topsSuccess, #topsErrorInvalidValue, #topsErrorMemoryAllocation
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
 *  gcu200 (pavo)   | NO
 *  gcu210 (dorado) | NO
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
TOPS_PUBLIC_API
topsError_t topsMemCacheInvalidate(topsDeviceptr_t dptr, size_t size);

/**
 * Invalidate all caches of device memory.
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | NO
 *  gcu210 (dorado) | NO
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | NO
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
TOPS_PUBLIC_API
topsError_t topsMemCacheInvalidateAll(topsStream_t stream);

/**
 * Flush all caches of device memory.
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | NO
 *  gcu210 (dorado) | NO
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.3_topsrider   | NO
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
TOPS_PUBLIC_API
topsError_t topsMemCacheFlushAll(topsStream_t stream);

/**
 * Get available memory size for mc
 *
 * @param[in] mc mc index.
 * @param[out] size Pointer to available size for this mc.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsExtGetMcAvailableMemSize(uint64_t mc, uint64_t *size);

/**
 * Get section count of the executable
 *
 * @param[in] exe Pointer to executable object.
 * @param[in] sh_type Section header type.
 * @param[out] count Pointer to count of the section.
 *
 * @return topsSuccess on success, or other on failure.
 *
 * Note: call with sh_type = topsExtExecutableSectionHeaderType::SHT_LAST_TYPE
 * will return total section counts of executable
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
TOPS_PUBLIC_API
topsError_t topsExecutableGetSectionCount(topsExecutable_t exe,
    topsExtExecutableSectionHeaderType_t sh_type, uint64_t *count);

/**
 * Get specific section information of the executable
 *
 * @param[in] exe Pointer to executable object.
 * @param[in] sh_type Section header type.
 * @param[in] mc mc index.
 * @param[out] info section information.
 *
 * @return topsSuccess on success, or other on failure.
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
TOPS_PUBLIC_API
topsError_t topsExecutableGetSectionInfo(topsExecutable_t exe,
    topsExtExecutableSectionHeaderType_t sh_type, int mc,
    topsExtExecutableSectionInfo_t *info);

/**
 * Get memory usage information.
 *
 * @param[out] current_used_bytes Current size of used memory in bytes.
 * @param[out] peak_used_bytes Peak size of used memory in bytes.
 * @param[in] bank_total The total number of memory bank.
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.4_topsrider   | YES
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
TOPS_PUBLIC_API
topsError_t topsGetMemUsageInfo(size_t *current_used_bytes,
                                size_t *peek_used_bytes,
                                int bank_total);

/**
 * Get memory bank list of affinity
 *
 * @param[out] affinity_bank_list Affinity bank list of memory.
 * @param[in] bank_total The total number of memory bank.
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.4_topsrider   | YES
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
TOPS_PUBLIC_API
topsError_t topsGetAffinityBankList(int *affinity_bank_list,
                                    int bank_total);
/**
 * Get memory bank list of affinity
 *
 * @param [in] res Pointer to res_bundle object.
 * @param[out] affinity_bank_list Affinity bank list of memory.
 * @param[in] bank_total The total number of memory bank.
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.4_topsrider   | YES
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
TOPS_PUBLIC_API
topsError_t topsGetAffinityBankListV2(topsResource_t res,
                                      int *affinity_bank_list,
                                      int bank_total);

/**
 * @brief Query memory usage info list.
 *
 * Return list of free, used, max available,
 * and total memory on the memory bank.
 *
 * @param [out] used_list Pointer to used memory list.
 * @param [out] free_list Pointer to free memory list.
 * @param [out] max_available_list Pointer to max_available memory list.
 * @param [out] total_list Pointer to total memory list.
 * @param[in] bank_total The total number of memory bank.
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
 * Interface          | -                        |
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
TOPS_PUBLIC_API
topsError_t topsMemGetInfoExt(size_t *used_list, size_t *free_list,
                              size_t *max_available_list, size_t *total_list,
                              int bank_total);

/* ECCL kernel signal api begin */
/**
 * get the maximum kernel signal number on current device.
 *
 * @param[out] num a pointer to the variable alloced by caller.
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.4_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsKernelSignalMaxNumGet(int *num);
/**
 * get the available kernel signal number on current device.
 *
 * @param[out] num a pointer to the variable alloced by caller.
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.4_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsKernelSignalAvailableNumGet(int *num);
/**
 * get a kernel signal handle alloced by kmd on current device.
 *
 * @param[out] handle put output of handle on here
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.4_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsKernelSignalAlloc(int *handle);
/**
 * free a kernel signal handle alloced by kmd on current device.
 *
 * @param[in] handle the kernel signal handle
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.4_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsKernelSignalFree(int handle);
/**
 * get a kernel signal value indicated by handle.
 *
 * @param[in]   handle the kernel signal handle
 * @param[out]  value  put output of value on here
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.4_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsKernelSignalRead(int handle, uint32_t *value);
/**
 * set a kernel signal value indicated by handle.
 *
 * @param[in] handle the kernel signal handle
 * @param[in] value  the value of kernel signal
 *
 * @return topsSuccess on success, or other on failure.
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
 *  gcu200 (pavo)   | YES
 *  gcu210 (dorado) | YES
 *  gcu300 (scorpio)| YES
 *
 *  software        | support
 * -------------    | -------------
 *  2.4_topsrider   | YES
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsKernelSignalWrite(int handle, uint32_t value);
/* ECCL kernel signal api end */
/**
 * create a roce sq.
 *
 * @param[in] port         the ROCE port id
 * @param[in] sq_base_ptr  the base address of the sq's ring buffer
 * @param[in] sq_size      the size of the sq's ring buffer
 * @param[in] sq_user      the user of the sq
 * @param[out] q_id        the allocated sq id
 *
 * @return topsSuccess on success, or other on failure.
 *
 * @cond INTERNAL
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
 *  gcu200 (pavo)   | NO
 *  gcu210 (dorado) | NO
 *  gcu300 (scorpio)| NO
 *  gcu400 (libra)  | YES
 *
 *  software        | support
 * -------------    | -------------
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3.2
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsRoceCreateQueue(uint32_t port, topsDeviceptr_t sq_base_ptr,
                             size_t sq_size, uint8_t sq_user, uint32_t *q_id);
/**
 * query the specified sq's info.
 *
 * @param[in] port  the ROCE port id
 * @param[in] q_id  the sq id
 * @param[out] mac  mac address binded with the sq
 * @param[out] id   ip address binded with the sq
 *
 * @return topsSuccess on success, or other on failure.
 *
 * @cond INTERNAL
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
 *  gcu200 (pavo)   | NO
 *  gcu210 (dorado) | NO
 *  gcu300 (scorpio)| NO
 *  gcu400 (libra)  | YES
 *
 *  software        | support
 * -------------    | -------------
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3.2
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsRoceQueryQueue(uint32_t port, uint32_t q_id, uint64_t *mac,
                               uint32_t *ip);
/**
 * Bind a queue pair.
 *
 * @param[in] port         the ROCE port id
 * @param[in] q_id         the local sq id
 * @param[in] remote_q_id  the remote sq id
 * @param[in] remote_mac   the remote mac address binded with the remote sq
 * @param[in] remote_ip    the remote ip address binded with the remote sq
 *
 * @return topsSuccess on success, or other on failure.
 *
 * @cond INTERNAL
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
 *  gcu200 (pavo)   | NO
 *  gcu210 (dorado) | NO
 *  gcu300 (scorpio)| NO
 *  gcu400 (libra)  | YES
 *
 *  software        | support
 * -------------    | -------------
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3.2
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsRoceBindQueuePair(uint32_t port, uint32_t q_id,
                                  uint32_t remote_q_id, uint64_t remote_mac,
                                  uint32_t remote_ip);
/**
 * Delete a sq.
 *
 * @param[in] port         the ROCE port id
 * @param[in] q_id         the local sq id
 *
 * @return topsSuccess on success, or other on failure.
 *
 * @cond INTERNAL
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
 *  gcu200 (pavo)   | NO
 *  gcu210 (dorado) | NO
 *  gcu300 (scorpio)| NO
 *  gcu400 (libra)  | YES
 *
 *  software        | support
 * -------------    | -------------
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3.2
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsRoceDeleteQueue(uint32_t port, uint32_t q_id);
/**
 * emit a write request to sq for master mode.
 *
 * @param[in] port         the ROCE port id
 * @param[in] q_id         the local sq id
 * @param[in] dst          destination device ptr
 * @param[in] src          source device ptr
 * @param[in] size         the write request size
 *
 * @return topsSuccess on success, or other on failure.
 *
 * @cond INTERNAL
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
 *  gcu200 (pavo)   | NO
 *  gcu210 (dorado) | NO
 *  gcu300 (scorpio)| NO
 *  gcu400 (libra)  | YES
 *
 *  software        | support
 * -------------    | -------------
 *
 * ####Dependency Analysis#####
 * Depend Module | Version
 * ------------- | -------------
 * KMD           | 97.3.2
 *
 * ####Benchmark Analysis#####
 *
 * @endcond
 */
TOPS_PUBLIC_API
topsError_t topsRoceWriteQueue(uint32_t port, uint32_t q_id, void *dst,
                                                     void *src, size_t size);
/**
 * @}
 */

#ifdef __cplusplus
} /* extern "c" */
#endif

#endif  // #iidef TOPS_INCLUDE_TOPS_TOPS_EXT_H

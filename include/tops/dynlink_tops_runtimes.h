/*
 * The confidential and proprietary information contained in this file may
 * only be used by a person authorised under and to the extent permitted
 * by a subsisting licensing agreement from Enflame Tech.Co., Ltd.
 *
 *            (C) COPYRIGHT 2022-2026 Enflame Tech.Co., Ltd.
 *                ALL RIGHTS RESERVED
 *
 * This entire notice must be reproduced on all copies of this file
 * and copies of this file may only be made by a person if such person is
 * permitted to do so under the terms of a subsisting license agreement
 * from Enflame Tech.Co., Ltd.
 *
 * Author: TOPSCODEC
 * Date: 2023.12.20
 */

#ifndef _DYNLINK_TOPS_RUNTIMES_H_
#define _DYNLINK_TOPS_RUNTIMES_H_

#include "tops/tops_runtime_api.h"

typedef topsError_t ttopsInit(unsigned int flags);
typedef topsError_t ttopsDriverGetVersion(int* driverVersion);
typedef topsError_t ttopsRuntimeGetVersion(int* runtimeVersion);
typedef topsError_t ttopsDeviceGet(topsDevice_t* device, int ordinal);
typedef topsError_t ttopsDeviceComputeCapability(int* major, int* minor, topsDevice_t device);
typedef topsError_t ttopsDeviceGetName(char* name, int len, topsDevice_t device);
typedef topsError_t ttopsDeviceGetPCIBusId(char* pciBusId, int len, int device);
typedef topsError_t ttopsDeviceGetByPCIBusId(int* device, const char* pciBusId);
typedef topsError_t ttopsDeviceTotalMem(size_t* bytes, topsDevice_t device);
typedef topsError_t ttopsDeviceSynchronize(void);
typedef topsError_t ttopsDeviceReset(void);
typedef topsError_t ttopsSetDevice(int deviceId);
typedef topsError_t ttopsGetDevice(int* deviceId);
typedef topsError_t ttopsGetDeviceCount(int* count);
typedef topsError_t ttopsDeviceGetAttribute(int* pi, topsDeviceAttribute_t attr, int deviceId);
typedef topsError_t ttopsGetDeviceProperties(topsDeviceProp_t* prop, int deviceId);
typedef topsError_t ttopsDeviceGetLimit(size_t* pValue, enum topsLimit_t limit);
typedef topsError_t ttopsGetDeviceFlags(unsigned int* flags);
typedef topsError_t ttopsSetDeviceFlags(unsigned flags);
typedef topsError_t ttopsChooseDevice(int* device, const topsDeviceProp_t* prop);
typedef topsError_t ttopsIpcGetMemHandle(topsIpcMemHandle_t* handle, void* devPtr);

typedef topsError_t ttopsGetLastError(void);
typedef topsError_t ttopsPeekAtLastError(void);
typedef const char* ttopsGetErrorName(topsError_t tops_error);
typedef const char* ttopsGetErrorString(topsError_t topsError);

typedef topsError_t ttopsPointerGetAttributes(topsPointerAttribute_t* attributes, const void* ptr);
typedef topsError_t ttopsPointerGetAttribute(void* data, topsPointer_attribute attribute,topsDeviceptr_t ptr);
typedef topsError_t ttopsDrvPointerGetAttributes(unsigned int numAttributes, topsPointer_attribute* attributes, void** data, topsDeviceptr_t ptr);
typedef topsError_t ttopsMalloc(void** ptr, size_t size);
typedef topsError_t ttopsExtCodecMemHandle(void** pointer, uint64_t dev_addr, size_t size);
typedef topsError_t ttopsExtMallocWithFlags(void** ptr, size_t sizeBytes, unsigned int flags);
typedef topsError_t ttopsHostMalloc(void** ptr, size_t size, unsigned int flags);
typedef topsError_t ttopsHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags);
typedef topsError_t ttopsHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
typedef topsError_t ttopsHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
typedef topsError_t ttopsHostUnregister(void* hostPtr);
typedef topsError_t ttopsFree(void* ptr);
typedef topsError_t ttopsHostFree(void* ptr);
typedef topsError_t ttopsMemcpy(void* dst, const void* src, size_t sizeBytes, topsMemcpyKind kind);
typedef topsError_t ttopsMemcpyWithStream(void* dst, const void* src, size_t sizeBytes,topsMemcpyKind kind, topsStream_t stream);
typedef topsError_t ttopsMemcpyHtoD(topsDeviceptr_t dst, void* src, size_t sizeBytes);
typedef topsError_t ttopsMemcpyDtoH(void* dst, topsDeviceptr_t src, size_t sizeBytes);
typedef topsError_t ttopsMemcpyDtoD(topsDeviceptr_t dst, topsDeviceptr_t src, size_t sizeBytes);
typedef topsError_t ttopsMemcpyHtoDAsync(topsDeviceptr_t dst, void* src, size_t sizeBytes, topsStream_t stream);
typedef topsError_t ttopsMemcpyDtoHAsync(void* dst, topsDeviceptr_t src, size_t sizeBytes, topsStream_t stream);
typedef topsError_t ttopsMemcpyDtoDAsync(topsDeviceptr_t dst, topsDeviceptr_t src, size_t sizeBytes,topsStream_t stream);
typedef topsError_t ttopsMemset(void* dst, int value, size_t sizeBytes);
                                                              
#endif //_DYNLINK_TOPS_RUNTIMES_H_
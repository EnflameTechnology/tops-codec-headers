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
#ifndef _DYNLINK_LOADER_H_
#define _DYNLINK_LOADER_H_

#include <stdlib.h>
#include <stdio.h>
#include <dlfcn.h>
#include "tops/dynlink_tops_codec.h"
#include "tops/dynlink_tops_runtimes.h"

#define TOPSCODEC_LIBNAME     "libtopscodec.so"
#define TOPS_RUNTIMES_LIBNAME "libtopsrt.so" //libefdrv.so libefrt.so librtcu.so

#define TOPS_LIB_HANDLE void*
#define TOPS_LOAD_FUNC(path) dlopen((path), RTLD_LAZY)
#define TOPS_SYM_FUNC(lib, sym) dlsym((lib), (sym))
#define TOPS_FREE_FUNC(lib) dlclose(lib)

enum debug_level {
  DEBUG_LEVEL_DISABLE = 0,
  DEBUG_LEVEL_ERR,
  DEBUG_LEVEL_INFO,
  DEBUG_LEVEL_DEBUG
};

#ifndef efdebug
#define efdebug  DEBUG_LEVEL_DEBUG
#endif

#define PRINT printf
#define TOPS_LOG_FUNC(...)                                                    \
  do {                                                                        \
    if (efdebug >= DEBUG_LEVEL_DEBUG) {                                       \
      PRINT(__VA_ARGS__);                                                     \
    }                                                                         \
  } while (0)

#define LOAD_LIBRARY(l, path)                                                 \
    do {                                                                      \
        if (!((l) = TOPS_LOAD_FUNC(path))) {                                  \
            TOPS_LOG_FUNC("Cannot load %s\n", path);                          \
            ret = -1;                                                         \
            goto error;                                                       \
        }                                                                     \
        TOPS_LOG_FUNC("Loaded lib: %s\n", path);                              \
    } while (0)

#define LOAD_SYMBOL(fun, tp, symbol)                                          \
    do {                                                                      \
        if (!((f->fun) = (tp*)TOPS_SYM_FUNC(f->lib, symbol))) {               \
            TOPS_LOG_FUNC("Cannot load %s\n", symbol);                        \
            ret = -1;                                                         \
            goto error;                                                       \
        }                                                                     \
        TOPS_LOG_FUNC("Loaded sym: %s\n", symbol);                            \
    } while (0)

#define LOAD_SYMBOL_OPT(fun, tp, symbol)                                      \
    do {                                                                      \
        if (!((f->fun) = (tp*)TOPS_SYM_FUNC(f->lib, symbol))) {               \
            TOPS_LOG_FUNC("Cannot load optional %s\n", symbol);               \
        } else {                                                              \
            TOPS_LOG_FUNC("Loaded sym: %s\n", symbol);                        \
        }                                                                     \
    } while (0)

#define GENERIC_LOAD_FUNC_PREAMBLE(T, n, N)                                   \
    T *f;                                                                     \
    int ret;                                                                  \
                                                                              \
    n##_free_functions(functions);                                            \
                                                                              \
    f = *functions = (T*)calloc(1, sizeof(*f));                               \
    if (!f)                                                                   \
        return -1;                                                            \
                                                                              \
    LOAD_LIBRARY(f->lib, N);

#define GENERIC_LOAD_FUNC_FINALE(n)                                           \
    return 0;                                                                 \
error:                                                                        \
    n##_free_functions(functions);                                            \
    return ret;

#define GENERIC_FREE_FUNC()                                                   \
    if (!functions)                                                           \
        return;                                                               \
    if (*functions && (*functions)->lib)                                      \
        TOPS_FREE_FUNC((*functions)->lib);                                    \
    free(*functions);                                                         \
    *functions = NULL;

/* topscodec function definition */
typedef struct TopsCodecFunctions_t {
    ttopscodecGetLibVersion   *lib_topscodecGetLibVersion;
    ttopscodecGetMemoryHandle *lib_topscodecGetMemoryHandle;
    ttopscodecDecGetCaps      *lib_topscodecDecGetCaps;
    ttopscodecDecCreate       *lib_topscodecDecCreate;
    ttopscodecDecSetParams    *lib_topscodecDecSetParams;
    ttopscodecDecDestroy      *lib_topscodecDecDestroy;
    ttopscodecDecodeStream    *lib_topscodecDecodeStream;
    ttopscodecDecFrameMap     *lib_topscodecDecFrameMap;
    ttopscodecDecFrameUnmap   *lib_topscodecDecFrameUnmap;
    ttopscodecGetLoading      *lib_topscodecGetLoading;

    TOPS_LIB_HANDLE lib;
} TopsCodecFunctions;

typedef struct TopsRuntimesFunctions_t {
    ttopsRuntimeGetVersion    *lib_topsRuntimeGetVersion;
    ttopsDeviceGet            *lib_topsDeviceGet;
    ttopsDeviceGetName        *lib_topsDeviceGetName;
    ttopsGetErrorName         *lib_topsGetErrorName;
    ttopsGetErrorString       *lib_topsGetErrorString;
    ttopsMalloc               *lib_topsMalloc;
    ttopsPointerGetAttributes *lib_topsPointerGetAttributes;
    ttopsPointerGetAttribute  *lib_topsPointerGetAttribute;
    ttopsHostMalloc           *lib_topsHostMalloc;
    ttopsHostGetDevicePointer *lib_topsHostGetDevicePointer;
    ttopsMemcpy               *lib_topsMemcpy;
    ttopsMemcpyHtoD           *lib_topsMemcpyHtoD;
    ttopsMemcpyDtoH           *lib_topsMemcpyDtoH;
    ttopsMemcpyDtoD           *lib_topsMemcpyDtoD;
    ttopsMemset               *lib_topsMemset;
    ttopsFree                 *lib_topsFree;
    ttopsHostFree             *lib_topsHostFree;

    TOPS_LIB_HANDLE lib;
} TopsRuntimesFunctions;

static void topscodec_free_functions(TopsCodecFunctions **functions)
{
    GENERIC_FREE_FUNC();
}

static void topsruntimes_free_functions(TopsRuntimesFunctions **functions)
{
    GENERIC_FREE_FUNC();
}

static int topscodec_load_functions(TopsCodecFunctions **functions)
{
    GENERIC_LOAD_FUNC_PREAMBLE(TopsCodecFunctions,
                                topscodec,
                                TOPSCODEC_LIBNAME);

    LOAD_SYMBOL(lib_topscodecGetLibVersion,   ttopscodecGetLibVersion,
                                             "topscodecGetLibVersion");
    LOAD_SYMBOL(lib_topscodecGetMemoryHandle, ttopscodecGetMemoryHandle,
                                             "topscodecGetMemoryHandle");
    LOAD_SYMBOL(lib_topscodecDecGetCaps,      ttopscodecDecGetCaps,
                                             "topscodecDecGetCaps");
    LOAD_SYMBOL(lib_topscodecDecCreate,       ttopscodecDecCreate,
                                             "topscodecDecCreate");
    LOAD_SYMBOL(lib_topscodecDecSetParams,    ttopscodecDecSetParams,
                                             "topscodecDecSetParams");
    LOAD_SYMBOL(lib_topscodecDecDestroy,      ttopscodecDecDestroy,
                                             "topscodecDecDestroy");
    LOAD_SYMBOL(lib_topscodecDecodeStream,    ttopscodecDecodeStream,
                                             "topscodecDecodeStream");
    LOAD_SYMBOL(lib_topscodecDecFrameMap,     ttopscodecDecFrameMap,
                                             "topscodecDecFrameMap");
    LOAD_SYMBOL(lib_topscodecDecFrameUnmap,   ttopscodecDecFrameUnmap,
                                             "topscodecDecFrameUnmap");
    //LOAD_SYMBOL(lib_topscodecGetLoading,      ttopscodecGetLoading,
    //                                         "topscodecGetLoading");
    GENERIC_LOAD_FUNC_FINALE(topscodec);
}

static inline int topsruntimes_load_functions(TopsRuntimesFunctions **functions)
{
    GENERIC_LOAD_FUNC_PREAMBLE(TopsRuntimesFunctions,
                                topsruntimes, 
                                TOPS_RUNTIMES_LIBNAME);

    LOAD_SYMBOL(lib_topsRuntimeGetVersion,    ttopsRuntimeGetVersion,
                                             "topsRuntimeGetVersion");
    LOAD_SYMBOL(lib_topsDeviceGet,            ttopsDeviceGet,
                                             "topsDeviceGet");
    LOAD_SYMBOL(lib_topsDeviceGetName,        ttopsDeviceGetName,
                                             "topsDeviceGetName");
    LOAD_SYMBOL(lib_topsGetErrorName,         ttopsGetErrorName,
                                             "topsGetErrorName");
    LOAD_SYMBOL(lib_topsGetErrorString,       ttopsGetErrorString,
                                             "topsGetErrorString");
    LOAD_SYMBOL(lib_topsMalloc,               ttopsMalloc,
                                             "topsMalloc");
    LOAD_SYMBOL(lib_topsPointerGetAttributes, ttopsPointerGetAttributes,
                                             "topsPointerGetAttributes");
    LOAD_SYMBOL(lib_topsPointerGetAttribute,  ttopsPointerGetAttribute,
                                             "topsPointerGetAttribute");
    LOAD_SYMBOL(lib_topsHostMalloc,           ttopsHostMalloc,
                                             "topsHostMalloc");
    LOAD_SYMBOL(lib_topsHostGetDevicePointer, ttopsHostGetDevicePointer,
                                             "topsHostGetDevicePointer");
    LOAD_SYMBOL(lib_topsMemcpy,               ttopsMemcpy,
                                             "topsMemcpy");
    LOAD_SYMBOL(lib_topsMemcpyHtoD,           ttopsMemcpyHtoD,
                                             "topsMemcpyHtoD");
    LOAD_SYMBOL(lib_topsMemcpyDtoH,           ttopsMemcpyDtoH,
                                             "topsMemcpyDtoH");
    LOAD_SYMBOL(lib_topsMemcpyDtoD,           ttopsMemcpyDtoD,
                                             "topsMemcpyDtoD");                                         
    LOAD_SYMBOL(lib_topsMemset,               ttopsMemset,
                                             "topsMemset");
    LOAD_SYMBOL(lib_topsFree,                 ttopsFree,
                                             "topsFree");
    LOAD_SYMBOL(lib_topsHostFree,             ttopsHostFree,
                                             "topsHostFree");
    GENERIC_LOAD_FUNC_FINALE(topsruntimes);
}

static inline int tops_runtimes_check(void *topsGetErrorName_fn, 
                                      void *topsGetErrorString_fn,
                                      topsError_t err, const char *func)
{
    const char *err_name;
    const char *err_string;

    TOPS_LOG_FUNC("Calling %s\n", func);

    if (err == topsSuccess)
        return 0;

    err_name = ((ttopsGetErrorName *)topsGetErrorName_fn)(err);
    err_string = ((ttopsGetErrorString *)topsGetErrorString_fn)(err);

    TOPS_LOG_FUNC("%s failed", func);
    if (err_name && err_string)
        TOPS_LOG_FUNC(" -> %s: %s", err_name, err_string);

    TOPS_LOG_FUNC("\n");
    return err;
}


#define TOPS_CHECK_LIB(topsl, x)                               \
            tops_runtimes_check(topsl->lib_topsGetErrorName,   \
                          topsl->lib_topsGetErrorString,       \
                          (x), #x)

#endif //_DYNLINK_LOADER_H_
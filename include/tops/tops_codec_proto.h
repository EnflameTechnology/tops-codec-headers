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

#ifndef _TOPS_CODEC_PROTO_H_
#define _TOPS_CODEC_PROTO_H_

#include "tops/tops_codec.h"

/*!
 * @brief Describe the warpper of api topscodecDecGetCaps.
 */
typedef struct{
    i32_t                       ret;
    topscodecType_t               codec;
    u32_t                       device_id;
    u32_t                       reserved;
    topscodecDecCaps_t            caps;
}topscodecDecGetCaps_t;

/*!
 * @brief Describe the warpper of api topscodecDecCreate.
 */
typedef struct{
    i32_t                       ret;
    topscodecHandle_t             handle;
    u64_t                       id;
    topscodecDecCreateInfo_t      info;
}topscodecDecCreate_t;

/*!
 * @brief Decribe the warpper of api topscodecDecSetParams.
 */
typedef struct{
    i32_t                       ret;
    topscodecHandle_t             handle;
    u64_t                       id;
    topscodecDecParams_t          params;
}topscodecDecSetParams_t;

/*!
 * @brief Decribe the warpper of api topscodecDecDestroy.
 */
typedef struct{
    i32_t                       ret;
    topscodecHandle_t             handle;
    u64_t                       id;
}topscodecDecDestory_t;

/*!
 * @brief Decribe the warpper of api topscodecDecodeStream.
 */
typedef struct{
    i32_t                       ret;
    topscodecHandle_t             handle;
    u64_t                       id;
    topscodecStream_t             input;
    i32_t                       timeout_ms;
}topscodecDecodeStream_t;

/*!
 * @brief Decribe the warpper of api topscodecDecFrameMap.
 */
typedef struct{
    i32_t                       ret;
    topscodecHandle_t             handle;
    u64_t                       id;
    topscodecFrame_t              frame;
}topscodecDecFrameMap_t;

/*!
 * @brief Decribe the warpper of api topscodecDecFrameUnmap.
 */
typedef struct{
    i32_t                       ret;
    topscodecHandle_t             handle;
    u64_t                       id;
    topscodecFrame_t              frame;
}topscodecDecFrameUnmap_t;

/*!
 * @brief Decribe the warpper of api topscodecDecGetStatus.
 */
typedef struct{
    i32_t                       ret;
    topscodecHandle_t             handle;
    u64_t                       id;
    topscodecDecStatus_t          status;
}topscodecDecGetStatus_t;

/*!
 * @brief Decribe the warpper of api topscodecGetLoading.
 */
typedef struct{
    i32_t                       ret;
    topscodecDevID_t              deviceID;
    double                      loading;
}topscodecGetLoading_t;


/*!
 * @brief Decribe the warpper of api topscodecGetLibVersion.
 */
typedef struct{
    i32_t                       ret;
    u32_t                       major;
    u32_t                       minor;
    u32_t                       patch;
}topscodecGetLibVersion_t;

#endif

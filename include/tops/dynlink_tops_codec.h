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

#ifndef _DYNLINK_TOPS_CODEC_H_
#define _DYNLINK_TOPS_CODEC_H_

#include "tops/tops_codec.h"

typedef  i32_t ttopscodecGetLibVersion(u32_t *major,u32_t *minor,u32_t *patch);
typedef  i32_t ttopscodecGetMemoryHandle(u64_t dev_addr,u64_t size, void** rt_handle);
typedef  i32_t ttopscodecDecGetCaps(topscodecType_t codec, u32_t card_id, u32_t device_id, topscodecDecCaps_t *caps);
typedef  i32_t ttopscodecDecCreate(topscodecHandle_t *handle, topscodecDecCreateInfo_t *info);
typedef  i32_t ttopscodecDecSetParams(topscodecHandle_t handle, topscodecDecParams_t *params);
typedef  i32_t ttopscodecDecDestroy(topscodecHandle_t handle);
typedef  i32_t ttopscodecDecodeStream(topscodecHandle_t handle, topscodecStream_t *input, i32_t timeout_ms);
typedef  i32_t ttopscodecDecFrameMap(topscodecHandle_t handle, topscodecFrame_t *frame);
typedef  i32_t ttopscodecDecFrameUnmap(topscodecHandle_t handle, topscodecFrame_t *frame);
typedef  i32_t ttopscodecDecGetStatus(topscodecHandle_t handle, topscodecDecStatus_t *status);
typedef  i32_t ttopscodecGetLoading(topscodecDevID_t deviceID, u32_t sess_id, double *loading);

#endif //_DYNLINK_TOPS_CODEC_H_
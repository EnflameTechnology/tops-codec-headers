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

#ifndef _TOPSCODEC_DEC_H_
#define _TOPSCODEC_DEC_H_

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#define TOPSCODEC_MAJOR_VERSION 1
#define TOPSCODEC_MINOR_VERSION 0
#define TOPSCODEC_PATCH_VERSION 0

#include <limits.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>

#include "tops/tops_codec_export.h"

#if defined(WIN32) || defined(WINDOWS)
typedef unsigned __int64 u64_t;
typedef unsigned __int32 u32_t;
typedef unsigned __int16 u16_t;
typedef unsigned __int8 u8_t;
typedef __int64 i64_t;
typedef signed __int32 i32_t;
typedef signed __int16 i16_t;
typedef signed __int8 i8_t;
typedef unsigned __int32 bool_t;
#else
typedef uint64_t u64_t;
typedef uint32_t u32_t;
typedef uint16_t u16_t;
typedef uint8_t u8_t;
typedef int64_t i64_t;
typedef int32_t i32_t;
typedef int16_t i16_t;
typedef int8_t i8_t;
typedef uint32_t bool_t;
#endif /*WIN32||WINDOWS*/

typedef void *topscodecHandle_t;
typedef uint32_t MVE_BUFFERTYPE;

#define OMX_MAX_STRINGNAME_SIZE 128

/*!
*@brief Describes VPP codec type
*/
typedef enum {
  TOPSCODEC_VPP_COMBO = 0,        /*!< Combo */   
  TOPSCODEC_VPP_DECODER_ONLY,     /*!< Decoder only */
  TOPSCODEC_VPP_UNKNOWN
} topscodecVppType_t;

/*!
*@brief Describes the device ID
*/
typedef enum {
  TOPSCODEC_DEVICE_VPP0_COMBO = 0,        /*!< Device 0*/   
  TOPSCODEC_DEVICE_VPP0_DECODE,           /*!< Device 1*/
  TOPSCODEC_DEVICE_VPP1_COMBO,            /*!< Device 2*/
  TOPSCODEC_DEVICE_VPP1_DECODE    ,       /*!< Device 3*/
  TOPSCODEC_DEVICE_UNKNOWN
} topscodecDevID_t;

/*!
 * @brief Describes the return values of EFCodec API calls.
 */
typedef enum {
  TOPSCODEC_SUCCESS = 0,
  TOPSCODEC_ERROR_INVALID_VALUE,      /*!< Invalid parameters passed to EFCodec. */
  TOPSCODEC_ERROR_INVALID_HANDLE,     /*!< Invalid handle passed to EFCodec. */
  TOPSCODEC_ERROR_INVALID_MEMORY,     /*!< Invalid memory passed to EFCodec. */
  TOPSCODEC_ERROR_CREATE_FAILED,      /*!< Fails to create encode or decode. */
  TOPSCODEC_ERROR_TIMEOUT,            /*!< Timeout for function call. */
  TOPSCODEC_ERROR_OUT_OF_MEMORY,      /*!< Unable to allocate enough memory. */
  TOPSCODEC_ERROR_BUFFER_EMPTY,       /*!< No output buffer available. */
  TOPSCODEC_ERROR_NOT_SUPPORTED,      /*!< Function not supported. */
  TOPSCODEC_ERROR_NOT_PERMITED,       /*!< Operation not permitted. */
  TOPSCODEC_ERROR_TRANSMIT_FAILED,    /*!< An error occurs in msg transmit process. */
  TOPSCODEC_ERROR_BAD_STREAM,         /*!< Invalid input stream; for example, it fails to parse the JPEG stream. */
  TOPSCODEC_ERROR_BUFFER_OVERFLOW,    /*!< Encode output buffer overflow. */
  TOPSCODEC_ERROR_DEVICE_NOT_OPEN, // Encoder Device has not open yet.
  TOPSCODEC_ERROR_UNKNOWN             /*!< Unknown error. */
} topscodecRetCode_t;

/*!
 * @brief Describes the codec types.
 */
typedef enum {
  TOPSCODEC_MPEG1 = 0,                    /*!<      MPEG1        */
  TOPSCODEC_MPEG2,                        /*!<      MPEG2        */
  TOPSCODEC_MPEG4,                        /*!<      MPEG4        */
  TOPSCODEC_VC1,                          /*!<      VC1          */
  TOPSCODEC_H263,                         /*!<      H263         */
  TOPSCODEC_H264,                         /*!<      H.264        */
  TOPSCODEC_H264_SVC,                     /*!<      H.264-SVC    */
  TOPSCODEC_H264_MVC,                     /*!<      H.264-MVC    */
  TOPSCODEC_HEVC,                         /*!<      HEVC         */
  TOPSCODEC_VP8,                          /*!<      VP8          */
  TOPSCODEC_VP9,                          /*!<      VP9          */
  TOPSCODEC_AVS,                          /*!<      AVS          */
  TOPSCODEC_AVS_PLUS,                     /*!<      AVS+         */
  TOPSCODEC_AVS2,                         /*!<      AVS2         */
  TOPSCODEC_JPEG,                         /*!<      JPEG         */
  TOPSCODEC_AV1,                          /*!<      AV1         */
  TOPSCODEC_NUM_CODECS                    /*!< The number of EFCodec enums. */
} topscodecType_t;

/*!
 * @brief Describes the EFCodec backend IP
 */
typedef enum {
  TOPSCODEC_BACKEND_DEFAULT_HW = 0,
  TOPSCODEC_BACKEND_JPU_HW = 1,
  TOPSCODEC_BACKEND_VPU_HW = 2,
} topscodecJpegBackend_t;

/*!
 * @brief Describes the chroma formats.
 */
typedef enum {
  TOPSCODEC_CHROMA_FORMAT_MONOCHROME = 0, /*!< MonoChrome. */
  TOPSCODEC_CHROMA_FORMAT_420,            /*!< YUV 4:2:0. */
  TOPSCODEC_CHROMA_FORMAT_422,            /*!< YUV 4:2:2. */
  TOPSCODEC_CHROMA_FORMAT_444,            /*!< YUV 4:4:4. */
  TOPSCODEC_CHROMA_FORMAT_440,            /*!< YUV 4:4:0. */
  TOPSCODEC_CHROMA_FORMAT_411,            /*!< YUV 4:1:1. */
  TOPSCODEC_CHROMA_FORMAT_410,            /*!< YUV 4:1:0. */
  TOPSCODEC_CHROMA_FORMAT_400,            /*!< YUV 4:0:0. */
  TOPSCODEC_CHROMA_FORMAT_UNKNOWN,        /*!< Unknown chroma format. */
} topscodecChromaFormat_t;

/*!
 * @brief Describes the picture types.
 */
typedef enum {
  TOPSCODEC_PIC_TYPE_P = 0,              /*!< Forward predicted picture. */
  TOPSCODEC_PIC_TYPE_B,                  /*!< Bi-directionally predicted picture. */
  TOPSCODEC_PIC_TYPE_I,                  /*!< Intra predicted picture. */
  TOPSCODEC_PIC_TYPE_IDR,                /*!< IDR picture. */
  TOPSCODEC_PIC_TYPE_UNKNOWN,            /*!< Unknown picture type. */
} topscodecPicType_t;

/*!
 * @brief Describes the stream types.
 */
typedef enum {
  TOPSCODEC_NALU_TYPE_P = 0,             /*!< Forward predicted frame type. */
  TOPSCODEC_NALU_TYPE_B,                 /*!< Bi-directionally predicted frame type. */
  TOPSCODEC_NALU_TYPE_I,                 /*!< Intra predicted frame type. */
  TOPSCODEC_NALU_TYPE_IDR,               /*!< IDR frame type.*/
  TOPSCODEC_NALU_TYPE_EOS,               /*!< EOS (End Of Stream) NAL unit type. */
  TOPSCODEC_NALU_TYPE_SEI,               /*!< SEI NAL unit type. */
  TOPSCODEC_NALU_TYPE_SPS,               /*!< SPS NAL unit type. */
  TOPSCODEC_NALU_TYPE_PPS,               /*!< PPS NAL unit type. */
  TOPSCODEC_NALU_TYPE_VPS,               /*!< VPS NAL unit type. */
  TOPSCODEC_H264_NALU_TYPE_SPS_PPS,      /*!< H.264 mixed type, not a standard NALU type. */
  TOPSCODEC_HEVC_NALU_TYPE_VPS_SPS_PPS,  /*!< HEVC mixed type, not a standard NALU type. */
  TOPSCODEC_NALU_TYPE_UNKNOWN,           /*!< Unknown NAL unit type. */
} topscodecStreamType_t;

/*!
 * @brief Describes the event types.
 */
typedef enum {
  TOPSCODEC_EVENT_NEW_FRAME = 0,         /*!< For both decode/encode, data output callback event. */
  TOPSCODEC_EVENT_SEQUENCE,              /*!< For video decode, sequence callback event. */
  TOPSCODEC_EVENT_EOS,                   /*!< For both decode/encode, notifies EOS event. */
  TOPSCODEC_EVENT_FRAME_PROCESSED,       /*!< Encode frame is processed. */
  TOPSCODEC_EVENT_BITSTREAM_PROCESSED,   /*!< Decode input bitstream buffer*/
  TOPSCODEC_EVENT_OUT_OF_MEMORY,         /*!< Fails to allocate memory due to insufficient space. */
  TOPSCODEC_EVENT_STREAM_CORRUPT,        /*!< Stream corrupts, and the frame is discarded. */
  TOPSCODEC_EVENT_STREAM_NOT_SUPPORTED,  /*!< Stream is not supported. */
  TOPSCODEC_EVENT_BUFFER_OVERFLOW,
  /*!< Encode output buffer overflow or decode output buffer number is not enough. */
  TOPSCODEC_EVENT_FATAL_ERROR,           /*!< Internal fatal error. */
} topscodecEventType_t;

/*!
 * @brief Describes the color space types.
 */
typedef enum {
  TOPSCODEC_COLOR_SPACE_BT_601 = 0,      /*!< ITU BT.601 color standard. */
  TOPSCODEC_COLOR_SPACE_BT_601_ER,       /*!< ITU BT.601 color standard extend range. */
  TOPSCODEC_COLOR_SPACE_BT_709,          /*!< ITU BT.709 color standard. */
  TOPSCODEC_COLOR_SPACE_BT_709_ER,       /*!< ITU BT.709 color standard extend range. */
  TOPSCODEC_COLOR_SPACE_BT_2020,         /*!< ITU BT.2020 color standard. */
  TOPSCODEC_COLOR_SPACE_BT_2020_ER,      /*!< ITU BT.2020 color standard extend range. */
} topscodecColorSpace_t;

/*!
 * @brief Describes the memory types.
 */
typedef enum {
  TOPSCODEC_MEM_TYPE_HOST = 0,           /*!< Host CPU memory. */
  TOPSCODEC_MEM_TYPE_DEV,                /*!< GCU Device memory. */
} topscodecMemType_t;

/*!
 * @brief Describes the buffer source types.
 */
typedef enum {
  TOPSCODEC_BUF_SOURCE_LIB = 0,          /*!< Creates buffers by EFCodec SDK. */
  TOPSCODEC_BUF_SOURCE_USER,             /*!< Creates buffers by user APP. */
} topscodecBufSource_t;

/*!
 * @brief Indicates frame plane attributes, which are used in ::topscodecFrame_t.
 */
typedef struct {
  u64_t dev_addr;                      /*!< Device memory address. */
  u32_t stride;                        /*!< Stride of picture. */
  u32_t alloc_len;                     /*!< Allocated buffer size of surface plane. */
} topscodecFramePlane_t;

/*************************************************************************************************************/
/*!
 * @brief Describes the pixel formats.
 */
typedef enum topscodecPixelFormat_t{
  TOPSCODEC_PIX_FMT_NV12 = 0,             /*!< Semi-planar Y4-U1V1.                  */
  TOPSCODEC_PIX_FMT_NV21,                 /*!< Semi-planar Y4-V1U1.                  */
  TOPSCODEC_PIX_FMT_I420,                 /*!< Planar Y4-U1-V1.                      */
  TOPSCODEC_PIX_FMT_YV12,                 /*!< Planar Y4-V1-U1.                      */
  TOPSCODEC_PIX_FMT_YUYV,                 /*!< 8bit packed Y2U1Y2V1.                 */
  TOPSCODEC_PIX_FMT_UYVY,                 /*!< 8bit packed U1Y2V1Y2.                 */
  TOPSCODEC_PIX_FMT_YVYU,                 /*!< 8bit packed Y2V1Y2U1.                 */
  TOPSCODEC_PIX_FMT_VYUY,                 /*!< 8bit packed V1Y2U1Y2.                 */
  TOPSCODEC_PIX_FMT_P010,                 /*!< 10bit semi-planar Y4-U1V1.            */
  TOPSCODEC_PIX_FMT_P010LE,               /*!< 10bit semi-planar Y4-U1V1 little end  */
  TOPSCODEC_PIX_FMT_I010,                 /*!< 10bit planar Y4-U1-V1.                */
  TOPSCODEC_PIX_FMT_YUV444,               /*!< 8bit planar Y4-U4-V4                  */
  TOPSCODEC_PIX_FMT_YUV444_10BIT,         /*!< 10bit planar Y4-U4-V4.                */
  TOPSCODEC_PIX_FMT_ARGB,                 /*!< Packed A8R8G8B8.                      */
  TOPSCODEC_PIX_FMT_ABGR,                 /*!< Packed A8B8G8R8.                      */
  TOPSCODEC_PIX_FMT_BGRA,                 /*!< Packed B8G8R8A8.                      */
  TOPSCODEC_PIX_FMT_RGBA,                 /*!< Packed A8B8G8R8.                      */
  TOPSCODEC_PIX_FMT_RGB565,               /*!< R5G6B5, 16 bits per pixel.            */
  TOPSCODEC_PIX_FMT_BGR565,               /*!< B5G6R5, 16 bits per pixel.            */
  TOPSCODEC_PIX_FMT_RGB555,               /*!< B5G5R5, 16 bits per pixel.            */
  TOPSCODEC_PIX_FMT_BGR555,               /*!< B5G5R5, 16 bits per pixel.            */
  TOPSCODEC_PIX_FMT_RGB444,               /*!< R4G4B4, 16 bits per pixel.            */
  TOPSCODEC_PIX_FMT_BGR444,               /*!< B4G4R4, 16 bits per pixel.            */
  TOPSCODEC_PIX_FMT_RGB888,               /*!< 8bit packed R8G8B8.                   */
  TOPSCODEC_PIX_FMT_BGR888,               /*!< 8bit packed B8G8R8.                   */
  TOPSCODEC_PIX_FMT_RGB3P,                /*!< 8bit planar R-G-B                     */
  TOPSCODEC_PIX_FMT_RGB101010,            /*!< 10bit packed R10G10B10.               */
  TOPSCODEC_PIX_FMT_BGR101010,            /*!< 10bit packed B10G10R10.               */
  TOPSCODEC_PIX_FMT_MONOCHROME,           /*!< 8bit gray scale.                      */
  TOPSCODEC_PIX_FMT_MONOCHROME_10BIT,     /*!< 10bit gray scale.                     */
  TOPSCODEC_PIX_FMT_BGR3P,                /*!< 8bit planar B-G-R                     */
} topscodecPixelFormat_t;
/*************************************************************************************************************************/

#define TOPSCODEC_FRAME_MAX_PLANE_NUM (6U)
/*!
 * @brief Indicates frame attributes, which are used as decoder's output data struct or encoder's input data struct.
 */
typedef struct {
  topscodecPixelFormat_t pixel_format;                      /*!< Specifies SurfaceFormat. */
  topscodecColorSpace_t color_space;                        /*!< Color standard. */
  u32_t width;                                            /*!< Specifies frame width in pixel. */
  u32_t height;                                           /*!< Specifies frame height in pixel. */
  u32_t device_id;                                        /*!< Specifies device ordinal. */
  u32_t plane_num;                                        /*!< The number of surface planes of current frame. */
  topscodecFramePlane_t plane[TOPSCODEC_FRAME_MAX_PLANE_NUM]; /*!< The information of each surface of current frame. */
  topscodecPicType_t pic_type;                              /*!< Specifies the picture type. */
  u32_t mem_channel;                                      /*!< Reserved. */
  u64_t pts;                                              /*!< Presentation timestamp. */
  u64_t priv_data;                                        /*!< Passes user information. Triggle Sync Decode, [0:Disable, 1:Enable]*/
} topscodecFrame_t;

/*!
 * @brief Indicates stream attributes, which are used as decoding input or encoding output.
 *        Only system memory is supported when this struct is used as decoding input.
 *        Only GCU memory is supported when this struct is used as encoding output.
 */
typedef struct {
  u64_t mem_addr;
  /*!< Stream memory address, GCU memory address or system memory address. */
  u32_t data_offset;                                      /*!< Valid data offset from mem_addr.*/
  u32_t alloc_len;                                        /*!< size of memo*/
  /*!< Stream buffer allocated size, which is set by buffer owner. */
  u32_t data_len;                                         /*!< Valid data size from mem_addr + data_offset，Valid data length .*/
  /*!< It is required that data_offset + data_len <= alloc_len. */
  topscodecMemType_t mem_type;                              /*!< Specifies memory type. */
  topscodecStreamType_t stream_type;
  /*!< Specifies stream type. */
  u64_t pts;                                              /*!< The presentation timestamp. */
  u64_t priv_data;                                        /*!< Passes user information. Triggle Sync Decode, [0:Disable, 1:Enable]*/
} topscodecStream_t;

/*!
 * @brief Indicates rectangle attributes, which are used to indicate CROP attribute or video decoding display area.
 */
typedef struct {
  u16_t x;                                                /*!< Top left corner of picture. */
  u16_t y;                                                /*!< Top left corner of picture. */
  u16_t width;                                            /*!< Width of picture. */
  u16_t height;                                           /*!< Height of picture. */
} topscodecRect_t;

/*!
 * @brief Indicates video decoding display aspect ratio.
 */
typedef struct {
  u32_t width;                                            /*!< Width of picture. */
  u32_t height;                                           /*!< Height of picture. */
} topscodecAspectRatio_t;

/*!
 * @brief Indicates video stream's framerate, numerator/denominator.
 */
typedef struct {
  u32_t numerator;                                        /*!< Numerator. */
  u32_t denominator;                                      /*!< Denominator. */
} topscodecFps_t;

/*!
 * @brief Video signal description. Refer to Section E.2.1 (VUI parameters syntax) of H.264 specification doc.
 */
typedef struct {
  u8_t video_format;             /*!< 0-Component, 1-PAL, 2-NTSC, 3-SECAM, 4-MAC, 5-Unspecified (not supported yet). */
  u8_t video_full_range_flag;    /*!< Indicates the black level, luma and chroma range. */
  u8_t color_primaries;          /*!< Chromaticity coordinates of source primaries. */
  u8_t transfer_characteristics; /*!< Opto-electronic transfer characteristic of the source picture. */
  u8_t matrix_coefficients;      /*!< Derives luma and chroma signals from RGB primaries. */
  u8_t reserved[3];              /*!< Reserved. */
} topscodecVideoSignalDescription_t;

/*!
 * @brief Indicates the rotation facotrs.
 */
typedef enum {
  TOPSCODEC_ROTATION_NONE = 0,   /*!< Counter-clockwise rotation 0. */
  TOPSCODEC_ROTATION_90   = 90,  /*!< Counter-clockwise rotation 90. */
  TOPSCODEC_ROTATION_180  = 180, /*!< Counter-clockwise rotation 180. */
  TOPSCODEC_ROTATION_270  = 270, /*!< Counter-clockwise rotation 270. */ 
}topscodecRotation_t;

/*!
 * @brief Indicates codec ratation attributes.
 */
typedef struct {
  u32_t                 enable;  /*!< Supports to rotation or not. */
  topscodecRotation_t   rotation;  /*!< Rotation angle*/
}topscodecRotationAttr_t;

/*!
 * @brief Indicates decoding downscaled width and height attributes.
 */
typedef struct {
  u32_t enable;           /*!< Supports to downscale or not. */
  u32_t interDslMode;     /*!< Downscale mode: 0-Bilinear, 1-Nearest*/
  u32_t width;            /*!< Width of picture. */
  u32_t height;           /*!< Height of picture. */
} topscodecDownscaleAttr_t;

/*!
 * @brief Indicates crop attributes.
 */
typedef struct {
  u32_t enable;                                          /*!< Supports to crop or not. */
  u32_t tl_x;                                            /*!< Top left corner of picture x. */
  u32_t tl_y;                                            /*!< Top left corner of picture y. */
  u32_t br_x;                                            /*!< Bottom right corner of picture x. */
  u32_t br_y;                                            /*!< Bottom right corner of picture y. */
} topscodecCropAttr_t;

/*
 * @brief Indicates output frame sampling attributes.
 */

typedef struct{
  u32_t enable;       /*!< Support to sample ouput frame or not*/
  u32_t sfo;          /*!< Frame sampling interval*/
  u32_t sf_idr;       /*!< IDR Frame sampling*/
}topscodecSfoAttr_t;

/*!
 * @brief Indicates decoding post-process attributes or
 *        encoding pre-process attributes.
 */
typedef struct {
  topscodecDownscaleAttr_t   downscale;      /*!< Scale attribute. */
  topscodecCropAttr_t        crop;           /*!< Crop attribute.*/
  topscodecRotationAttr_t    rotation;       /*!< Rotation attribute*/
  topscodecSfoAttr_t         sf;             /*!< Selected ouput frame*/
} topscodecPpAttr_t;

/*!
 * @brief Describes the run mode, of which synchronized mode is only supported
 *        by JPEG decoding and JPEG encoding.
 */
typedef enum {
  TOPSCODEC_RUN_MODE_ASYNC = 0,            /*!< Runs asynchronously. */
  TOPSCODEC_RUN_MODE_SYNC,                 /*!< Runs synchronously. Only JPEG supports this mode. */
} topscodecRunMode_t;

/**
 * @brief Decoder and encoder callback function type, which is called when events of topscodecEventType_t occur.
 *
 * @details The first parameter with type topscodecEventType_t indicates event type.
 *          The second parameter with type void * is used to bypass user_context.
 *          The third parameter with type void * indicates event attribute.
 *          Return value of callbacks will be passed to SendData when an error occurs.
 */
typedef i32_t (*topscodecCallback_t)(topscodecEventType_t, void *, void *);

/*!
 * @brief Retrieves the version of EFCodec library. The version of EFCodec is composed
 * of \b major, \b minor and \b patch. For instance, major = 1, minor = 1, patch = 9,
 * the version of EFCodec library is 1.1.9.
 *
 * @param[in] major
 * Input. A pointer to scale factor that gets the major version of EFCodec library.
 * @param[in] minor
 * Input. A pointer to scale factor that gets the minor version of EFCodec library.
 * @param[in] patch
 * Input. A pointer to scale factor that gets the patch version of EFCodec library.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the value
 *         of the function parameter is invalid. 
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern TOPSCODEC_EXPORT i32_t topscodecGetLibVersion(u32_t *major, u32_t *minor, u32_t *patch);


//! @brief Describes video decoding mode.
typedef enum {
  TOPSCODEC_DEC_MODE_IPB = 0,   /*!< Decodes IPB frames. */
  TOPSCODEC_DEC_MODE_IP,        /*!< Decodes IP frames (not supported yet). */
  TOPSCODEC_DEC_MODE_I,         /*!< Decodes I frames. */
  TOPSCODEC_DEC_MODE_REF,       /*!< Decodes reference frames, and skips non-reference frames. */
} topscodecDecMode_t;

//! @brief Describes decoder sending modes.
typedef enum {
  TOPSCODEC_DEC_SEND_MODE_FRAME = 0,    /*!< Sends by frame. */
  TOPSCODEC_DEC_SEND_MODE_STREAM,       /*!< Sends by stream. */
} topscodecDecSendMode_t;

//! @brief Describes video decoding output order.
typedef enum {
  TOPSCODEC_DEC_OUTPUT_ORDER_DISPLAY = 0, /*!< Display order. */
  TOPSCODEC_DEC_OUTPUT_ORDER_DECODE,      /*!< Decoding order. */
} topscodecDecOutputOrder_t;

/*!
 * @brief Describes the decoder capability. The capability of JPEG decoder is a union of all backends.
 */
typedef struct {
  u8_t   supported;               /*!< Indicates whether the decoder is supported. */
  u8_t   reserved1[3];            /*!< Reserved. */
  u32_t  max_width;               /*!< The maximum width supported. */
  u32_t  max_height;              /*!< The maximum height supported. */
  u32_t  min_width;               /*!< The minimum width supported. */
  u32_t  min_height;              /*!< The minimum height supported. */
  u32_t  output_pixel_format_mask;/*!< Bit[n]:pixel format index supported. */
  u8_t   scale_up_supported;      /*!< Indicates the hardware support for scaling up. */
  u8_t   scale_down_supported;    /*!< Indicates the hardware support for scaling down. */
  u8_t   crop_supported;          /*!< Indicates the hardware support for cropping. */
  u8_t   rotation_supported;      /*!< Indicates the hardware support for rotation. */
  u8_t   resize_supported;      /*!< Indicates the hardware support for resiaze. */
  u8_t   reserved2[16];           /*!< Reserved. */
} topscodecDecCaps_t;

/*!
 * @brief Describes video decoding sequence event information.
 *
 */
typedef struct {
  topscodecType_t           codec;              /*!< Indicates the codec type. */
  u32_t                   coded_width;        /*!< Indicates coded frame width, in pixel. */
  u32_t                   coded_height;       /*!< Indicates coded frame height, in pixel. */
  u32_t                   min_output_buf_num;
  /*!< Indicates the minimum output buffer required for decoding,
       including reference framebuffer and extra framebuffer. */
  topscodecRect_t           display_area;       /*!< Indicates display crop area. */
  topscodecAspectRatio_t    aspect_ratio;       /*!< Indicates display aspect ratio. */
  topscodecFps_t            fps;                /*!< Indicates stream's framerate (Numerator/Denominator). */
  u32_t                   bit_depth;          /*!< Indicates the bit depth of stream. */
  u32_t                   interlaced;
  /*!< Indicates whether the stream is interlaced, 0: progressive, 1: interlaced. */
  topscodecChromaFormat_t   chroma_format;      /*!< Indicates chroma format. */
  topscodecVideoSignalDescription_t video_signal_description; /*!< Indicates video signal description. */
  u8_t reserved[8];                           /*!< Reserved. */
} topscodecDecSequenceInfo_t;

/*!
 * @brief Describes decoding stream corrupt event information.
 *
 */
typedef struct {
  u64_t   pts;            /*!< Presentation timestamp of decode frame. */
  u64_t   priv_data;      /*!< The user information of the decode frame. */
  u64_t   total_num;      /*!< Total presentation stream corrupt event number of the decoder. */
  u8_t    reserved[16];   /*!< Reserved. */
} topscodecDecStreamCorruptInfo_t;

/*!
 * @brief Describes decoder create attributes, which are used in ::topscodecDecCreate.
 */
typedef struct {
  u32_t                    device_id;         /*!< Specifies the device ID to use. */
  u32_t                    hw_ctx_id;         /*!< Hardware Context ID>*/
  u32_t                    sw_ctx_id;         /*!< Software Context ID*/
  u32_t                    session_id;        /*!< Host Pfofile Session ID*/
  topscodecType_t            codec;             /*!< Specifies the codec type. */
  topscodecDecSendMode_t     send_mode;         /*!< Specifies the decode send mode (not supported yet). */
  topscodecRunMode_t         run_mode;          /*!< Specifies the decode run mode. */
  u32_t                    stream_buf_size;
  /*!< Specifies the input stream buffer size; recommended value: w*h*1.25,
   * range [4096, device max memory size]. */
  u64_t                    user_context;      /*!< Bypasses user pointer. */

  topscodecJpegBackend_t     backend;
  /*!< Specifies the hardware backend for JPEG (the default is JPU hardware). */
  u8_t                     reserved[28];      /*!< Reserved and the value must be set to 0. */
} topscodecDecCreateInfo_t;

/*!
 * @brief Describes decoder parameters, which are used in ::topscodecDecSetParams.
 */
typedef struct {
  u32_t                    max_width;         /*!< Specifies the maximum width of the decoder, in pixel. */
  u32_t                    max_height;        /*!< Specifies the maximum video height of the decoder, in pixel. */
  u32_t                    stride_align;
  /*!< Specifies output picture stride alignment. The value is power of 2 in the range [1, 2048]. */
  u32_t                    output_buf_num;
  /*!< Specifies output buffer number, it is recommended to set to equal batchsize in sync mode
   * (sequence minimum output buffer number + app pipeline required buffer number). */
  i32_t                    mem_channel;       /*!< Reserved and the value must be set to 0. */
  topscodecPixelFormat_t     pixel_format;      /*!< Specifies output pixel format. */
  topscodecColorSpace_t      color_space;       /*!< Color transform matrix. */
  topscodecDecMode_t         dec_mode;          /*!< Specifies video decode mode (not supported yet). */
  topscodecDecOutputOrder_t  output_order;      /*!< Specifies the video decode output order. */
  topscodecBufSource_t       output_buf_source;
  /*!<  Lib mode or user mode (User buffer source is only supported in JPEG synchronization mode). */
  topscodecPpAttr_t          pp_attr;           /*!< Specifies the pre-process attribute. */
  u8_t                     reserved[32];      /*!< Reserved and the defalut value must be set to 0. [0: jpeg sync decode mode]*/
}  topscodecDecParams_t;

/*!
 * @brief Describes decoder buffer status, which is used in ::topscodecDecQueryBufStatus.
 */
typedef enum {
    TOPSCODECDEC_STATUS_INVALID         = 0,   // topscodec decode status is not valid
    TOPSCODECDEC_STATUS_INPROGRESS      = 1,   // topscodec decode is in progress
    TOPSCODECDEC_STATUS_SUCCESS         = 2,   // topscodec decode is completed without any errors
    // 3 to 7 enums are reserved for future use
    TOPSCODECDEC_STATUS_ERROR          = 8,    // topscodec decode is completed with an error
} topscodecDecStatus_t;

/*!
 * @brief Describes JPEG header information, which is used in ::topscodecDecGetJpegInfo.
 */
typedef struct {
  u32_t                   width;               /*!< Indicates JPEG picture width. */
  u32_t                   height;              /*!< Indicates JPEG picture height. */
  u32_t                   components_num;      /*!< Indicates JPEG picture component number. */
  topscodecChromaFormat_t   chroma_format;       /*!< Indicates JPEG picture chroma format. */
  bool                    hw_support;
} topscodecJpegInfo_t;


/*!
 * @brief Retrieves the version of EFCodec library. The version of EFCodec is
 * composed of major, minor and patch. For instance, major = 1, minor =
 * 1, patch = 9, the version of EFCodec library is 1.1.9.
 *
 * @param[in] major
 * Input. A pointer to scale factor that gets the major version of EFCodec
 * library.
 * @param[in] minor
 * Input. A pointer to scale factor that gets the minor version of EFCodec
 * library.
 * @param[in] patch
 * Input. A pointer to scale factor that gets the patch version of EFCodec
 * library.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the
 * value of the function parameter is invalid.
 * @note
 * - None.
 *
 * @par Requirements
 * - None.
 *
 * @par Example
 * - None.
 */
extern TOPSCODEC_EXPORT i32_t topscodecGetLibVersion(u32_t *major, u32_t *minor,
                                                  u32_t *patch);

/*!
  * @brief Convert a device address to a runtime memory handle.
  *
  * @param[in] dev_addr
  *   Specifies the device address.
  * @param[in] size
  *   Specifies the size of the memory.
  * @param[out] rt_handle
  *   Returns the runtime memory handle.
  *
  * @retval TOPSCODEC_SUCCESS: This function has run successfully.
  * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the
  * value of the function parameter is invalid.
  *
  */
extern TOPSCODEC_EXPORT i32_t topscodecGetMemoryHandle(u64_t dev_addr,
                                                    u64_t size,
                                                    void** rt_handle);

/*!
 * @brief Gets decoder capability information.
 *
 * @param[in] codec
 *   Specifies which codec is chosen to get capability information.
 * @param[in] device_id
 *   Specifies which device is chosen to get capability information. The value is in the range [0, device number - 1].
 * @param[out] caps
 *   Decoder's capability.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the value
 *         of the function parameter is invalid. 
 * @retval TOPSCODEC_ERROR_TRANSMIT_FAILED: This function call fails because communication  
 *         with the device fails.
 *   
 */
extern TOPSCODEC_EXPORT i32_t topscodecDecGetCaps(topscodecType_t codec, u32_t card_id, u32_t device_id, topscodecDecCaps_t *caps);

/*!
 * @brief Creates decoder.
 *
 * @param[out] handle
 *   Returns decoder's handle.
 * @param[in] cb
 *   Gives event callback to decoder.
 * @param[in] info
 *   Information for creating decoder.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the value
 *         of the function parameter is invalid.
 * @retval TOPSCODEC_ERROR_NOT_SUPPORTED: This function call fails because the codec type
 *          is not supported.
 * @retval TOPSCODEC_ERROR_CREATE_FAILED: This function call fails because it is unable to
 *         create decoder on the device.
 * @retval TOPSCODEC_ERROR_OUT_OF_MEMORY: This function call fails because it is unable to
 *         allocate enough memory.
 * @retval TOPSCODEC_ERROR_TRANSMIT_FAILED: This function call fails because communication 
 *         with the device fails.
 *   
 */
extern  TOPSCODEC_EXPORT i32_t topscodecDecCreate(topscodecHandle_t *handle, topscodecDecCreateInfo_t *info);

/*!
 * @brief Sets the decoder parameters.
 *
 * @param[in] handle
 *   Decoder's handle.
 * @param[in] params
 *   Decoder parameters.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle is invalid.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the value 
 *         of the function parameter is invalid.
 * @retval TOPSCODEC_ERROR_NOT_SUPPORTED: This function call fails because one or more 
 *         specified functions is not supported.
 * @retval TOPSCODEC_ERROR_OUT_OF_MEMORY: This function call fails because it is unable to 
 *         allocate enough memory.
 * @retval TOPSCODEC_ERROR_TRANSMIT_FAILED: This function call fails because communication 
 *         with the device fails.
 *
 * @note
 * - This function should be called before decoding or in the sequence 
 *   callback function.
 *
 * - When this function is called and it causes the output buffer to reallocate, the application needs
 *   to ensure that the old buffers are not referenced.
 */
extern TOPSCODEC_EXPORT i32_t topscodecDecSetParams(topscodecHandle_t handle, topscodecDecParams_t *params);

/*!
 * @brief Destroys decoder.
 *
 * @param[in] handle
 *   Decoder's handle.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle is invalid. 
 * @retval TOPSCODEC_ERROR_TRANSMIT_FAILED: This function call fails because communication
 *         with the device fails. 
 *   
 */
extern TOPSCODEC_EXPORT i32_t topscodecDecDestroy(topscodecHandle_t handle);

/*!
 * @brief Sends stream to decoder.
 *
 * @param[in] handle
 *   Decoder's handle.
 * @param[in] input
 *   Input stream information.
 * @param[in] timeout_ms
 *   Timeout in ms, -1 means waiting infinitely.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle is invalid.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the value 
 *         of the function parameter is invalid. 
 * @retval TOPSCODEC_ERROR_TRANSMIT_FAILED: This function call fails because communication
 *         with the device fails.
 * @retval TOPSCODEC_ERROR_TIMEOUT: This function call fails because of timeout. 
 * @retval TOPSCODEC_ERROR_UNKNOWN: This function call fails because of unknown error. The
 *         decoder is unable to continue running.
 *
 * @note
 * - When user's APP callback does not return TOPSCODEC_SUCCESS, the error return value will
 *   be passed to this function asynchronously.
 */
extern TOPSCODEC_EXPORT i32_t topscodecDecodeStream(topscodecHandle_t handle,
                                                 topscodecStream_t *input,
                                                 i32_t timeout_ms);

/*!
 * @brief Increases reference count of output frame.
 *
 * @param[in] handle
 *   Decoder's handle.
 * @param[in] frame
 *   Reference count of frame to be increased.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle
 *         is invalid.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the value 
 *         of the function parameter is invalid. 
 *   
 */
extern TOPSCODEC_EXPORT i32_t topscodecDecFrameMap(topscodecHandle_t handle, topscodecFrame_t *frame);

/*!
 * @brief Decreases reference count of output frame.
 *
 * @param[in] handle
 *   Decoder's handle.
 * @param[in] frame
 *   Reference count of frame to be decreased.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle is invalid.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the value
 *         of the function parameter is invalid.       
 * @retval TOPSCODEC_ERROR_TRANSMIT_FAILED: This function call fails because communication
 *         with the device fails.
 *   
 */
extern TOPSCODEC_EXPORT i32_t topscodecDecFrameUnmap(topscodecHandle_t handle, topscodecFrame_t *frame);

/*!
 * @brief Decodes JPEG by synchronization mode.
 *
 * @param[in] handle
 *   Decoder's handle.
 * @param[in] input
 *   Input stream information.
 * @param[in] output
 *   Output frame information, user application should allocate output frame buffer and set to output.
 * @param[in] timeout_ms
 *   Timeout in ms, -1 means waiting infinitely.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle is invalid.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the value
 *         of the function parameter is invalid.
 * @retval TOPSCODEC_ERROR_TIMEOUT: This function call fails because of timeout.
 * @retval TOPSCODEC_ERROR_BAD_STREAM: This function call fails because the input stream is invalid.
 * @retval TOPSCODEC_ERROR_NOT_SUPPORTED: This function call fails because the input stream is not supported.
 * @retval TOPSCODEC_ERROR_UNKNOWN: This function call fails because of unknown error.The 
 *         decoder is unable to continue running. 
 *   
 *   
 */
extern TOPSCODEC_EXPORT i32_t topscodecDecJpegSyncDecode(topscodecHandle_t handle,
                                                      topscodecStream_t input[],
                                                      topscodecFrame_t output[],
                                                      i32_t timeout_ms);


/*!
 * @brief Queries decoder's buffer status.
 *
 * @param[in] handle
 *   Decoder's handle.
 * @param[out] status
 *   Decoder's buffer status.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle is invalid.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the value
 *         of the function parameter is invalid. 
 * @retval TOPSCODEC_ERROR_TRANSMIT_FAILED: This function call fails because communication with the device fails.
 *   
 */
extern  TOPSCODEC_EXPORT i32_t topscodecDecGetStatus(topscodecHandle_t handle, topscodecDecStatus_t *status);

/*!
 * @brief Query hardware loading.
 *
 * @param[in] deviceID
 *   Codec device ID.
 * @param[in] Loading
 *   Codec HW Loading.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_VALUE: This function call fails because the value
 *         of the function parameter is invalid. 
 *   
 */
extern TOPSCODEC_EXPORT i32_t topscodecGetLoading(topscodecDevID_t deviceID, u32_t sess_id, double *loading);

/*!
 * @brief Set VPU Session Profiler.
 * 
 * @param[in] handle
 *   Codec's handle
 * @param[in] enable
 *   Flags to enable perf profiling.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle is invalid.
 *   
 */
// extern TOPSCODEC_EXPORT i32_t topscodecSetStreamProfiler(topscodecHandle_t pHandle, bool enable);

/*!
 * @brief Get VPU Session Profiler Status.
 * 
 * @param[in] handle
 *   Codec's handle
 * @param[out] enable
 *   Flags to enable perf profiling.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle is invalid.
 *   
 */
// extern TOPSCODEC_EXPORT i32_t topscodecGetStreamProfiler(topscodecHandle_t pHandle, bool *enable);

/*!
 * @brief Set VPU Device Profiler.
 * 
 * @param[in] handle
 *   Codec's handle
 * @param[in] enable
 *   Flags to enable perf profiling.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle is invalid.
 *   
 */
// extern TOPSCODEC_EXPORT i32_t topscodecSetDevProfiler(topscodecDevID_t device, bool enable);

/*!
 * @brief Get VPU Device Profiler Status.
 * 
 * @param[in] handle
 *   Codec's handle
 * @param[out] enable
 *   Flags to enable perf profiling.
 *
 * @retval TOPSCODEC_SUCCESS: This function has run successfully.
 * @retval TOPSCODEC_ERROR_INVALID_HANDLE: This function call fails because the handle is invalid.
 *   
 */
// extern TOPSCODEC_EXPORT i32_t topscodecGetDevProfiler(topscodecDevID_t device, bool *enable);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif  // INCLUDE_TOPSCODEC_V1_DEC_H_

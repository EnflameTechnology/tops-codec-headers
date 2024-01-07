include_guard(GLOBAL)

# ###############################################################################
# #########                      topscodec header                     ###########
# ###############################################################################
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/_cmake_ffmpeg_gcu_header)
set(FFMPEG_GCU_HEADER_PATH "${OUTPUT_DIR}/include" CACHE STRING "" FORCE)
set(FFMPEG_GCU_LIB_PATH "${OUTPUT_DIR}/lib" CACHE STRING "" FORCE)

set(ffmpeg_gcu_header
    ${OUTPUT_DIR}/include/tops/driver_types.h
    ${OUTPUT_DIR}/include/tops/dynlink_tops_codec.h
    ${OUTPUT_DIR}/include/tops/dynlink_tops_loader.h
    ${OUTPUT_DIR}/include/tops/dynlink_tops_runtimes.h
    ${OUTPUT_DIR}/include/tops/host_defines.h
    ${OUTPUT_DIR}/include/tops/tops_codec_export.h
    ${OUTPUT_DIR}/include/tops/tops_codec_proto.h
    ${OUTPUT_DIR}/include/tops/tops_codec.h
    ${OUTPUT_DIR}/include/tops/tops_common.h
    ${OUTPUT_DIR}/include/tops/tops_runtime_api.h
    ${OUTPUT_DIR}/include/tops/tops_version.h
    ${OUTPUT_DIR}/lib/pkgconfig/tops.pc
)

add_custom_command(
    OUTPUT ${ffmpeg_gcu_header}
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/ && make PREFIX=${OUTPUT_DIR} install
    USES_TERMINAL
)

add_custom_target(ffmpeg_gcu_header_target ALL
    DEPENDS ${ffmpeg_gcu_header}
)

include_directories(${FFMPEG_GCU_HEADER_PATH})
message("ffmpeg gcu header: " ${FFMPEG_GCU_HEADER_PATH})
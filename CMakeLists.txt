include_guard(GLOBAL)

# ###############################################################################
# #########                      topscodec header                     ###########
# ###############################################################################
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/_cmake_ffmpeg_gcu_header)
set(FFMPEG_GCU_HEADER_PATH "${OUTPUT_DIR}/include" CACHE STRING "" FORCE)
set(FFMPEG_GCU_LIB_PATH "${OUTPUT_DIR}/lib" CACHE STRING "" FORCE)

file(GLOB ffmpeg_gcu_header_srcs ${CMAKE_CURRENT_SOURCE_DIR}/include/tops/*.h)
list(APPEND ffmpeg_gcu_header_srcs ${CMAKE_CURRENT_SOURCE_DIR}/tops.pc.in)
set(ffmpeg_gcu_header_outs)

foreach(ffmpeg_gcu_header_src IN LISTS ffmpeg_gcu_header_srcs)
	string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "${OUTPUT_DIR}" ffmpeg_gcu_header_out_string "${ffmpeg_gcu_header_src}")
    if("${ffmpeg_gcu_header_src}" MATCHES "${CMAKE_CURRENT_SOURCE_DIR}/tops.pc.in")
        string(REPLACE "tops.pc.in" "tops.pc" ffmpeg_gcu_header_mod_out_string "${ffmpeg_gcu_header_src}")
        string(REPLACE "${CMAKE_CURRENT_SOURCE_DIR}" "${OUTPUT_DIR}/lib/pkgconfig" ffmpeg_gcu_header_out_string "${ffmpeg_gcu_header_mod_out_string}")
	list(APPEND ffmpeg_gcu_header_outs ${ffmpeg_gcu_header_mod_out_string})
    endif()
    list(APPEND ffmpeg_gcu_header_outs ${ffmpeg_gcu_header_out_string})
endforeach()

add_custom_command(
    OUTPUT ${ffmpeg_gcu_header_outs}
    COMMAND cd ${CMAKE_CURRENT_SOURCE_DIR}/ && make PREFIX=${OUTPUT_DIR} install
    # COMMAND touch -c ${ffmpeg_gcu_header}
    USES_TERMINAL
    DEPENDS ${ffmpeg_gcu_header_srcs}
)

add_custom_target(ffmpeg_gcu_header_target ALL
    DEPENDS ${ffmpeg_gcu_header_outs}
)

include_directories(${FFMPEG_GCU_HEADER_PATH})
message("ffmpeg gcu header: " ${FFMPEG_GCU_HEADER_PATH})

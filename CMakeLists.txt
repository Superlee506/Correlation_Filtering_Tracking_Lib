#
# Created by super on 18-7-16.
#
project(CfTracking)

cmake_minimum_required(VERSION 2.8.7)

# option to enable OpenMP; only relevant for the KCF version with the
# VOT scale estimation
option(WITH_OPENMP "Enable OpenMP" OFF)

if(WITH_OPENMP)
    find_package(OpenMP REQUIRED)

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif(WITH_OPENMP)

# add c++11 support
if(CMAKE_COMPILER_IS_GNUCC)
    ADD_DEFINITIONS ( -std=c++11 )
endif(CMAKE_COMPILER_IS_GNUCC)

# add OpenCV
set(OPENCV_DIR_HINT "")

if(WIN32)
    get_filename_component(OPENCV_DIR_PLATFORM $ENV{OPENCV_DIR} DIRECTORY)
    get_filename_component(OPENCV_DIR_HINT ${OPENCV_DIR_PLATFORM} DIRECTORY)
endif(WIN32)

set(OpenCV_STATIC OFF)
find_package(OpenCV REQUIRED HINTS ${OPENCV_DIR_HINT})

# add 3rdparty source
set(CF_TCLAP_DIR "src/3rdparty/tclap")
set(CF_CV_EXT_DIR "src/3rdparty/cv_ext")
set(CF_PIOTR_DIR "src/3rdparty/piotr")
set(CF_TRACKER_RUN "src/main/tracker_run")

include_directories(${OpenCV_DIR}/include)
include_directories(${CF_TCLAP_DIR}
                    ${CF_CV_EXT_DIR}
                    ${CF_PIOTR_DIR}
                    ${CF_PIOTR_DIR}/src
                    ${CF_TRACKER_RUN})

include_directories(src/cf_libs/kcf
                    src/cf_libs/dsst
                    src/cf_libs/common
                    src/cf_libs/staple)

set(CF_PIOTR_SOURCES
        ${CF_PIOTR_DIR}/gradientMex.hpp
        ${CF_PIOTR_DIR}/src/gradientMex.cpp
        ${CF_PIOTR_DIR}/src/sse.hpp
        ${CF_PIOTR_DIR}/src/wrappers.hpp)

set(CF_LIB_COMMON_SOURCES
        src/cf_libs/common/feature_channels.hpp
        src/cf_libs/common/mat_consts.hpp
        src/cf_libs/common/math_helper.hpp
        src/cf_libs/common/math_helper.cpp
        src/cf_libs/common/cf_tracker.hpp
        src/cf_libs/common/scale_estimator.hpp
        src/cf_libs/common/cv_ext.hpp
        src/cf_libs/common/tracker_debug.hpp
        src/cf_libs/common/image_acquisition.hpp
        src/cf_libs/common/image_acquisition.cpp
        ${CF_CV_EXT_DIR}/shift.cpp
        ${CF_CV_EXT_DIR}/shift.hpp
        ${CF_CV_EXT_DIR}/math_spectrums.cpp
        ${CF_CV_EXT_DIR}/math_spectrums.hpp
        ${CF_CV_EXT_DIR}/psr.hpp
        ${CF_PIOTR_SOURCES})

set(CF_MAIN_SOURCES
        ${CF_CV_EXT_DIR}/init_box_selector.hpp
        ${CF_CV_EXT_DIR}/init_box_selector.cpp
        ${CF_TRACKER_RUN}/tracker_run.hpp
        ${CF_TRACKER_RUN}/tracker_run.cpp
        ${CF_TCLAP_DIR}/tclap/CmdLine.h)

add_executable(STAPLE_run
        src/main/main_staple.cpp
        src/cf_libs/staple/staple_debug.hpp
        src/cf_libs/staple/colour_tracker.hpp
        src/cf_libs/staple/colour_tracker.cpp
        src/cf_libs/staple/staple_tracker.hpp
        src/cf_libs/staple/staple_tracker.cpp
        ${CF_MAIN_SOURCES}
        ${CF_LIB_COMMON_SOURCES})
target_link_libraries(STAPLE_run ${OpenCV_LIBS})

add_executable(DSST_run
        src/main/main_dsst.cpp
        src/cf_libs/dsst/dsst_debug.hpp
        src/cf_libs/dsst/dsst_tracker.hpp
        ${CF_MAIN_SOURCES}
        ${CF_LIB_COMMON_SOURCES})
target_link_libraries(DSST_run ${OpenCV_LIBS})

#add_executable(FDSST_run
#        src/main/main_fdsst.cpp
#        src/cf_libs/fdsst/fdsst_debug.hpp
#        src/cf_libs/fdsst/fdsst_tracker.hpp
#        ${CF_MAIN_SOURCES}
#        ${CF_LIB_COMMON_SOURCES})
#target_link_libraries(FDSST_run ${OpenCV_LIBS})

add_executable(KCF_run
        src/main/main_kcf.cpp
        src/cf_libs/kcf/kcf_debug.hpp
        src/cf_libs/kcf/kcf_tracker.hpp
        ${CF_MAIN_SOURCES}
        ${CF_LIB_COMMON_SOURCES})
target_link_libraries(KCF_run ${OpenCV_LIBS})

add_executable(Staple_Evaluate
        src/main/evaluate_staple.cpp
        src/cf_libs/staple/staple_debug.hpp
        src/cf_libs/staple/colour_tracker.hpp
        src/cf_libs/staple/colour_tracker.cpp
        src/cf_libs/staple/staple_tracker.hpp
        src/cf_libs/staple/staple_tracker.cpp
        ${CF_MAIN_SOURCES}
        ${CF_LIB_COMMON_SOURCES})
target_link_libraries(Staple_Evaluate ${OpenCV_LIBS})
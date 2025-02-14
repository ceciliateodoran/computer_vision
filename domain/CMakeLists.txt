cmake_minimum_required(VERSION 3.10)

project(domain)

# Set the OpenCV_DIR to the correct path
#set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
# or
if(UNIX)
    # Per sistemi Unix-like (Linux, macOS)
    set(OpenCV_DIR /home/cecilia/build)
# elseif(WIN32)
    # Per sistemi Windows
    # set(OpenCV_DIR C:/Users/*/Desktop/opencv/build)
else()
    message(FATAL_ERROR "Sistema operativo non supportato: compilare il programma c++ tramite docker")
endif()

find_package(OpenCV REQUIRED)

# Websocket
find_package(websocketpp REQUIRED)

# Boost
find_package(Boost REQUIRED)

# Aggiungi le directory di inclusione
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

set(CPM_DOWNLOAD_VERSION 0.38.1)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")

if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake...")
    file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
    )
endif()

include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
    NAME clustering
    GITHUB_REPOSITORY Lallapallooza/clustering
    GIT_TAG 0.0.1 # Choose tag
    OPTIONS "CLUSTERING_USE_AVX2 ON"
)

add_executable(${PROJECT_NAME} src/main/cpp/app.cpp)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} websocketpp::websocketpp Boost::boost clustering_header_lib)

# Installazione dell'eseguibile
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

# Installazione delle librerie OpenCV
file(GLOB OPENCV_LIBS "/home/cecilia/build/lib/libopencv_*.so*")
install(FILES ${OPENCV_LIBS} DESTINATION lib)
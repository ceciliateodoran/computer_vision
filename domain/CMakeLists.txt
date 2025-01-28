cmake_minimum_required(VERSION 3.10)

project(domain)

# compilazione statica

set(BUILD_SHARED_LIBS OFF)

set(OpenCV_STATIC ON)

if(UNIX)

    set(OpenCV_DIR /usr/local/share/opencv4)

else()

    message(FATAL_ERROR "Sistema operativo non supportato: compilare il programma c++ tramite docker")

endif()

# versioni statiche delle librerie

find_package(OpenCV REQUIRED)

find_package(websocketpp REQUIRED)

find_package(Boost REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} src/main/cpp/app.cpp)

# flag per il linking statico

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static")

# Link con le versioni statiche delle librerie

target_link_libraries(${PROJECT_NAME}

    -Wl,--start-group

    ${OpenCV_LIBS}

    websocketpp::websocketpp

    Boost::boost

    -static-libgcc

    -static-libstdc++

    -Wl,--end-group

)

# Modifica l'installazione per non copiare le librerie condivise

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)

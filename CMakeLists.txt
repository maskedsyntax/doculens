cmake_minimum_required(VERSION 3.10)
project(DocuLens)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(document_scanner
    src/main.cpp
    src/document_scanner.cpp
)

target_link_libraries(document_scanner
    ${OpenCV_LIBS}
)

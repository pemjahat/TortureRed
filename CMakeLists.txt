cmake_minimum_required(VERSION 3.20)
project(TortureRed VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/Build)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Enable FetchContent
include(FetchContent)

# Download and configure SDL2
FetchContent_Declare(
    SDL2
    URL https://github.com/libsdl-org/SDL/releases/download/release-2.30.0/SDL2-2.30.0.zip
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/SDL2
)

# Configure SDL2 build options
set(SDL_STATIC ON CACHE BOOL "" FORCE)
set(SDL_SHARED ON CACHE BOOL "" FORCE)
set(SDL_TEST OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(SDL2)

# Download DirectX Headers
FetchContent_Declare(
    directx_headers
    GIT_REPOSITORY https://github.com/microsoft/DirectX-Headers.git
    GIT_TAG v1.614.0
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/DirectX-Headers
)
FetchContent_MakeAvailable(directx_headers)

# Download and configure DXC (DirectX Shader Compiler)
FetchContent_Declare(
    dxc
    URL https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.7.2308/dxc_2023_08_14.zip
    SOURCE_DIR ${CMAKE_SOURCE_DIR}/ThirdParty/DXC
)
FetchContent_MakeAvailable(dxc)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/Sources)
include_directories(${sdl2_SOURCE_DIR}/include)
include_directories(${directx_headers_SOURCE_DIR}/include)

# Collect source files
file(GLOB_RECURSE SOURCES
    "${CMAKE_SOURCE_DIR}/Sources/*.cpp"
    "${CMAKE_SOURCE_DIR}/Sources/*.h"
    "${CMAKE_SOURCE_DIR}/Sources/*.hpp"
)

# Collect shader files separately (don't include in SOURCES to avoid MSBuild compilation)
file(GLOB_RECURSE SHADER_SOURCES
    "${CMAKE_SOURCE_DIR}/Sources/Shaders/*.hlsl"
)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    SDL2::SDL2
    SDL2::SDL2main
    d3d12.lib
    dxgi.lib
    dxguid.lib
    dxcompiler.lib
)

# Copy shader source files to output directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/Sources/Shaders
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/Shaders
)

# Create hardlinks for DLLs to Bin directory (symlinks require admin privileges on Windows)
if(WIN32)
    # Remove existing DLLs first to avoid conflicts
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E remove $<TARGET_FILE_DIR:${PROJECT_NAME}>/SDL2.dll
        COMMAND ${CMAKE_COMMAND} -E remove $<TARGET_FILE_DIR:${PROJECT_NAME}>/dxcompiler.dll
        COMMAND ${CMAKE_COMMAND} -E remove $<TARGET_FILE_DIR:${PROJECT_NAME}>/dxil.dll
    )

    # Create hardlinks using PowerShell (works without admin privileges)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND powershell -Command "New-Item -ItemType HardLink -Path '$<TARGET_FILE_DIR:${PROJECT_NAME}>/SDL2.dll' -Target '$<TARGET_FILE:SDL2::SDL2>' -Force"
    )

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND powershell -Command "New-Item -ItemType HardLink -Path '$<TARGET_FILE_DIR:${PROJECT_NAME}>/dxcompiler.dll' -Target '${dxc_SOURCE_DIR}/bin/x64/dxcompiler.dll' -Force"
    )

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND powershell -Command "New-Item -ItemType HardLink -Path '$<TARGET_FILE_DIR:${PROJECT_NAME}>/dxil.dll' -Target '${dxc_SOURCE_DIR}/bin/x64/dxil.dll' -Force"
    )
endif()

# Windows specific settings
if(WIN32)
    # Link additional Windows libraries needed for DirectX
    target_link_libraries(${PROJECT_NAME}
        d3dcompiler.lib
    )
endif()

# Copy content folder to output directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/Content
    $<TARGET_FILE_DIR:${PROJECT_NAME}>/Content
)

# Compiler warnings
if(MSVC)
    target_compile_options(${PROJECT_NAME} PRIVATE /W4)
else()
    target_compile_options(${PROJECT_NAME} PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Print configuration summary
message(STATUS "=== TortureRed Build Configuration ===")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Dependencies: SDL2, DirectX Headers, DXC (automatically downloaded)")
message(STATUS "Windowing: SDL2")
message(STATUS "Rendering: DirectX 12")
message(STATUS "ThirdParty Directory: ${CMAKE_SOURCE_DIR}/ThirdParty")
message(STATUS "Output Directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Runtime Shader Compilation: Triangle.hlsl -> VS/PS shaders (SM 6.5)")
message(STATUS "=====================================")
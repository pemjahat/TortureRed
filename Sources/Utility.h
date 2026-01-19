#pragma once

#include <iostream>
#include <cassert>

// Error checking macro for DirectX calls
#define CHECK_HR(hr, msg) \
    if (FAILED(hr)) { \
        std::cerr << "DirectX Error: " << msg << " (HRESULT: 0x" << std::hex << hr << ")" << std::endl; \
        assert(false); \
    }

#define CHECK_BOOL(condition, msg) \
    if (!(condition)) { \
        std::cerr << "Application Error: " << msg << std::endl; \
        assert(false); \
    }
#pragma once

// --- Enable microprofile ---
// Comment out (or set to 0) to completely compile-out all profiling (zero overhead)
#define MICROPROFILE_ENABLED 1

// --- GPU support ---
#define MICROPROFILE_GPU_TIMERS_D3D12 1

// --- Web server ---
#define MICROPROFILE_WEBSERVER 1
#define MICROPROFILE_WEBSERVER_PORT 1338
#define MICROPROFILE_WEBSERVER_MAXFRAMES 512

// --- Context switch tracing (Windows only) ---
#define MICROPROFILE_CONTEXT_SWITCH_TRACE 0

// --- Worker thread names ---
#define MICROPROFILE_THREAD_NAME_LEN 32

// --- Keybindings for the in-app overlay ---
#define MICROPROFILE_KEY_TOGGLE VK_F2
#define MICROPROFILE_KEY_MODE   VK_F1

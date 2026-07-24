#pragma once

// Quick And Easy GPU Random Numbers In D3D11 - Nathan Reed - 2013
// http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
// Hash Functions for GPU Rendering - Nathan Reed
// https://www.reedbeta.com/blog/hash-functions-for-gpu-rendering/

uint SeedThread(uint seed)
{
    uint state = seed * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

uint SeedThread(uint2 pixel, uint resolutionX, uint frameIndex)
{
    uint rngState = (pixel.y * resolutionX + pixel.x) ^ SeedThread(frameIndex);
    return SeedThread(rngState);
}

uint XORShift(inout uint rng_state)
{
    // Xorshift algorithm from George Marsaglia's paper
    rng_state ^= (rng_state << 13);
    rng_state ^= (rng_state >> 17);
    rng_state ^= (rng_state << 5);
    return rng_state;
}

float Random01(inout uint rng_state)
{
    return asfloat(0x3f800000 | XORShift(rng_state) >> 9) - 1.0;
}

float3 RandomColor(inout uint rng_state)
{
    return float3(Random01(rng_state), Random01(rng_state), Random01(rng_state));
}

uint RandomRange(inout uint rng_state, uint minimum, uint maximum)
{
    return minimum + uint(float(maximum - minimum + 1) * Random01(rng_state));
}

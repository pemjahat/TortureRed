#ifndef SHARC_INTEGRATION_HLSL
#define SHARC_INTEGRATION_HLSL

bool IsSharcQueryValid(float3 queryPosition, float hitDistance, float pathRoughness, SharcParameters sharcParams)
{
    uint gridLevel = HashGridGetLevel(queryPosition, sharcParams.gridParameters);
    float voxelSize = HashGridGetVoxelSize(gridLevel, sharcParams.gridParameters);
    bool isValidHit = hitDistance > voxelSize * sqrt(3.0f);

    float roughness = min(max(pathRoughness, 0.0f), 0.99f);
    float alpha = roughness * roughness;
    float footprint = hitDistance * sqrt(0.5f * alpha * alpha / max(1.0f - alpha * alpha, 1e-6f));

    return isValidHit && footprint > voxelSize;
}

#endif
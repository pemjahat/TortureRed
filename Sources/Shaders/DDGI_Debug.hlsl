// =============================================================================
// DDGI_Debug.hlsl — probe sphere overlay (post-composite LDR).
//
// Three views (FrameConstants::ddgiDebugView):
//   1 Irradiance — one billboard sphere for each active probe. Each sphere
//                  point samples the irradiance atlas along its own normal and
//                  undoes the gamma-5 curve, so the ball shows the directional
//                  structure the fetch will read. Scaled by ddgiDebugScale.
//   2 Probe states — the same spheres, with INACTIVE probes painted magenta on
//                  the camera-facing side. An inactive probe sits inside
//                  geometry: view 2 is the relocation check.
//   3 Depth — the same spheres, grayscale from the depth-moment atlas
//                  (M1 / (3 x spacing)). A probe in open space shows a bright
//                  ball; a probe against a wall shows a dark one.
//
// Post-composite: drawn after the TAA resolve and the tonemap directly onto the
// backbuffer at output resolution — display-referred colours, no temporal
// history to pollute. Occlusion is a manual reverse-Z compare against the
// GBuffer depth SRV. The geometry comes from SV_VertexID / SV_InstanceID with
// no vertex buffer.
// =============================================================================
#include "CommonTracing.hlsl"

ConstantBuffer<FrameConstants> FrameCB : register(b0);

// DDGI.hlsli includes BakedGI.hlsli, which references FrameCB.
#include "DDGI.hlsli"

#define DDGI_DEBUG_IRRADIANCE 1u
#define DDGI_DEBUG_STATES     2u
#define DDGI_DEBUG_DEPTH      3u

// UV-sphere tessellation — the layout BakedGI_Debug.hlsl and
// IrCache_DebugSpheres use (8 x 8 quads, 6 vertices for each quad).
#define DDGI_DBG_N_STACKS    8
#define DDGI_DBG_N_SLICES    8
#define DDGI_DBG_SPHERE_VERTS (DDGI_DBG_N_STACKS * DDGI_DBG_N_SLICES * 6)

static const uint kSphereCorner[6] = { 0u, 1u, 2u, 1u, 3u, 2u };

struct VSOut
{
    float4 pos        : SV_POSITION;
    float3 sphereN    : NORMAL;
    float2 uvRef      : TEXCOORD0; // placeholder to keep the semantic list explicit
    uint   probeIndex : PROBEINDEX;
};

VSOut VSMain(uint vid : SV_VertexID, uint instance : SV_InstanceID)
{
    VSOut o;
    o.pos        = float4(0.0f, 0.0f, -1.0f, 1.0f); // degenerate: culled
    o.sphereN    = float3(0.0f, 1.0f, 0.0f);
    o.uvRef      = float2(0.0f, 0.0f);
    o.probeIndex = instance;

    const DDGIVolumeGpu vol = DDGIGetVolumeFromFrame();
    if (instance >= vol.probeCount)
        return o;

    // Unit-sphere position: it doubles as the direction to sample the atlas at.
    const uint  v        = vid % DDGI_DBG_SPHERE_VERTS;
    const uint  quadIdx  = v / 6u;
    const uint  localIdx = v % 6u;
    const uint  stackIdx = quadIdx / DDGI_DBG_N_SLICES;
    const uint  sliceIdx = quadIdx % DDGI_DBG_N_SLICES;
    const uint  corner   = kSphereCorner[localIdx];
    const uint  si       = stackIdx + (corner >> 1u);
    const uint  li       = sliceIdx + (corner & 1u);
    const float sph      = (float)si / (float)DDGI_DBG_N_STACKS * PI;        // [0, PI]
    const float theta    = (float)li / (float)DDGI_DBG_N_SLICES * 2.0f * PI; // [0, 2PI]
    const float3 n = float3(sin(sph) * cos(theta), cos(sph), sin(sph) * sin(theta));

    const uint3 cell  = DDGIUnflatten3D(instance, vol.dims);
    const float3 base = vol.gridMin + (float3)cell * vol.spacing;

    StructuredBuffer<float4> offsets = ResourceDescriptorHeap[vol.probeOffsetSRVIdx];
    const float3 probePos = base + offsets[instance].xyz;

    o.sphereN = n;
    o.pos = mul(float4(probePos + n * (vol.spacing * 0.12f), 1.0f), FrameCB.viewProjUnjittered);
    return o;
}

float4 PSMain(VSOut i) : SV_Target
{
    // Manual occlusion against the GBuffer depth: map this output-resolution
    // pixel to its internal-resolution texel the way NaiveTsr_Resolve does.
    // NDC z is unaffected by the xy jitter, so the comparison is exact. This
    // engine's projection stores SV_Position.z = near / distance, which is the
    // same encoding as the GBuffer depth texture (nearer = larger).
    Texture2D<float> depthTex = ResourceDescriptorHeap[FrameCB.depthIndex];
    const float2 scale = float2(FrameCB.internalWidth, FrameCB.internalHeight)
                       / float2(FrameCB.outputWidth, FrameCB.outputHeight);
    const int2 ip = min(int2(i.pos.xy * scale),
                        int2(FrameCB.internalWidth, FrameCB.internalHeight) - 1);
    clip(i.pos.z - depthTex[ip] + 1e-5f); // visible when the ball is nearer

    const DDGIVolumeGpu vol = DDGIGetVolumeFromFrame();
    const uint view = FrameCB.ddgiDebugView;

    float3 colour = float3(0.0f, 0.0f, 0.0f);

    if (view == DDGI_DEBUG_DEPTH)
    {
        Texture2D<float2> depthMap = ResourceDescriptorHeap[vol.depthSRVIdx];
        const float2 uv = DDGIGetProbeUV(i.probeIndex, vol.probesPerRow, DDGI_PROBE_DEPTH_TEXELS,
                                         vol.depthTexSize, normalize(i.sphereN));
        // M1 / (3 x spacing): the same normalisation the relocation pass uses
        // for its "a frontface inside three probe sizes" rule.
        const float m1 = depthMap.SampleLevel(g_LinearSampler, uv, 0).x;
        colour = saturate(m1 / max(vol.spacing * 3.0f, 1e-6f)).xxx;
    }
    else
    {
        Texture2D<float4> irradiance = ResourceDescriptorHeap[vol.irradianceSRVIdx];
        const float2 uv = DDGIGetProbeUV(i.probeIndex, vol.probesPerRow, DDGI_PROBE_IRRADIANCE_TEXELS,
                                         vol.irradianceTexSize, normalize(i.sphereN));
        // Undo the gamma-5 encode: the stored value is pow(e, 1/5), and a ball
        // point shows e / 2pi (the resolve stores a hemisphere average).
        const float3 stored = irradiance.SampleLevel(g_LinearSampler, uv, 0).rgb;
        colour = pow(stored, DDGI_PROBE_GAMMA) / (2.0f * PI) * FrameCB.ddgiDebugScale;
    }

    // View 2 marks the probes the state resolve deactivated — the ones a
    // relocation could not push out of geometry.
    if (view == DDGI_DEBUG_STATES)
    {
        const uint3 cell = DDGIUnflatten3D(i.probeIndex, vol.dims);
        StructuredBuffer<float4> offsets = ResourceDescriptorHeap[vol.probeOffsetSRVIdx];
        const float3 probePos = vol.gridMin + (float3)cell * vol.spacing + offsets[i.probeIndex].xyz;
        const float3 toCamera = normalize(FrameCB.cameraPosition.xyz - probePos);
        if (!DDGIIsProbeActive(vol, i.probeIndex) && dot(toCamera, normalize(i.sphereN)) < 0.4f)
            colour = float3(1.0f, 0.0f, 1.0f);
    }

    return float4(saturate(colour), 1.0f);
}

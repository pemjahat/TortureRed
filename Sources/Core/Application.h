#pragma once

#include "Utility.h"
#include "Camera.h"
#include "Model.h"
#include "Rendering/Renderer.h"
#include "RenderDocCapture.h"

#include <vector>


#include "Graphics/Scene.h"

class Application
{
public:
    Application();
    ~Application();
    
    // ...
    
    Scene m_Scene;

    void Run();

private:
    void Initialize();
    void Shutdown();
    void ProcessEvents();
    void Update(float deltaTime);
    void Render();

    void InitializeImGui();
    void RenderImGui();
    void RenderRenderDocSection();

    // Fills the ray-debug FrameConstants from the live BakedGI getters. Called
    // from Application::Update and again from Application::Render right after
    // the dump, so the two fill sites cannot drift. It does NOT touch the sun
    // fold: FillSunFoldConstants owns that, and the post-dump call must not
    // re-derive a pick the dump did not change.
    void FillRayDebugConstants();

    // Publishes the two per-bake-term response balls the "Efb Contribution" view
    // (view 8) differences: the selected term's and term 1's (the no-feedback
    // baseline). Called from Application::Update and again from
    // Application::Render right after "Rebuild GI Probes", since the bake
    // replaces the balls and Update ran before it. BakedGI::kEfbBallNoSRV until
    // a bake has folded the terms.
    void FillEfbBallConstants();

    // Publishes the sun fold for the frame: the 16 normalised kernel weights
    // (FrameConstants::bakedGISunWeight, read by the Ray Fan's DeltaDirections
    // channel), the cached argmax and its weight (the Ray Fan's direction, and
    // the UI text), and the to-sun direction itself
    // (FrameConstants::bakedGISunDir, the 17th box that same channel draws). All of
    // it comes from one BakedGI::SunBlendForHour, so the channel, the UI and the
    // panel all describe one computation.
    void FillSunFoldConstants();

    bool m_IsRunning;
    bool m_EnableDepthPrePass    = false;
    bool m_DebugShadowMap        = false;
    bool m_UsePathTracer          = false;
    bool m_ShowDemoWindow         = false;
    bool m_UseMeshlet              = true;   // Toggle meshlet vs vertex-buffer rendering
    bool m_UseVisibilityBuffer    = true;   // Toggle Visibility Buffer (MeshletRasterizeMS token + VisibilityGBuffer resolve) vs direct-to-GBuffer (MeshletRasterizeGBufferMS)
    bool m_EnableTwoPassCulling = true;     // Toggle two-phase culling
    bool m_EnableOcclusionCulling = true;   // Toggle HZB occlusion culling
    bool m_FreezeCulling           = false;  // Freeze the culling view to visually verify instance/meshlet culling
    bool m_FPSLimitEnabled         = true;   // Enable FPS limiter
    float m_TargetFPS              = 60.0f;  // Target frames per second
    int  m_HZBDebugMip             = -1;     // -1=Off, else HZB mip visualized as grayscale
    bool m_OccludedRectDebug       = false;  // Draw NDC rects of HZB-occluded instances/meshlets
    bool m_DebugScreenText         = false;  // GPU on-screen debug text/lines overlay, master switch
    bool m_DebugDepthLabels        = false;  // Depth-duel labels for occluded records
    bool m_CullStatsOverlay        = false;  // Meshlet culling stats on-screen table
    float m_BakedGISpacing      = 1.0f;  // BakedGI probe grid spacing (m) — used at the next rebuild
    bool  m_PendingGIProbeBake  = false; // BakedGI bake request (auto-set after scene load / Rebuild button)
    int   m_BakedGIDebugView    = 0;     // Probe debug overlay: 0 off, 1 placement, 2 lit irradiance, 3 sky visibility,
                                         // 4 visibility rays, 5 lit sky source, 6 lit sun source, 7 ray fan,
                                         // 8 Efb contribution
    float m_BakedGIDebugScale   = 1.0f;  // Lit-ball gain (DC-normalized shading multiplier)
    int   m_BakedGIEfbIter      = 2;     // Efb contribution (view 8) bake term, 0-based: 0 = term 1 (no feedback ->
                                         // black), 1 = term 2, 2 = term 3 (the last, converged term). The UI shows
                                         // 1..kBakeIterations; the view sums nothing, it only picks which stored
                                         // response ball is the "selected" one (the baseline is always term 1).
    int   m_BakedGIEfbDir       = -1;    // Efb contribution (view 8) fold direction: -1 = fold all 16 directions
                                         // (the ball), 0..15 = that delta direction ALONE. A single direction
                                         // then reads the TRANSPORT TABLES instead of the folded balls, so a
                                         // tinted direction is not averaged against 15 neutral ones. The combo
                                         // index is the direction + 1, so entry 0 is "All"; the fill helper
                                         // clamps anything outside -1..15 back to "All".
    int   m_BakedGIRayDebugChannel = 1;  // Ray Fan (view 7) colour channel: 0 status, 1 albedo, 2 NdotD,
                                         // 3 Vhit, 4 Efb, 5 distance, 6 direct-vs-Efb, 7 DeltaDirections.
                                         // Channel 7 swaps the geometry to the 16 delta directions of the fan's
                                         // probe (grayscale, white = most weighted), adds one orange marker box
                                         // for the real sun, and reads no record.
                                         // BakedGI::kRayFanChannelDeltaDirs is the C++ spelling of that 7.
    float m_BakedGIRayDebugLength  = 8.0f; // Ray Fan ray length cap, in probe spacings (channels 0..6 only)
    int   m_BakedGIRayDebugRay     = -1;   // Cage Trace (view 9) gather-ray OVERRIDE: -1 = the shader's own
                                         // argmax (largest feedback-dominated ray), else the RECORD INDEX
                                         // (0..63) of the ray to trace, picked out of the pool the dump
                                         // built — the rays where direct < Efb (BakedGI::GetRayCandidates).
                                         // The combo's rows are that pool, so the pick is always a ray the
                                         // rule admits. It survives a dump on purpose: the index is
                                         // per-direction, so it keeps tracking one ray while the probe
                                         // changes. A pick from another direction's pool (the sun moved) is
                                         // drawn in magenta and reported in the panel.
                                         // BakedGI::kNoRayPick is the published spelling of -1.
    int   m_BakedGISunState     = 0;     // The first of the two hour states the fold
                                         // interpolates (BakedGI::SunHourBlend::i0).
                                         // Refreshed each frame by
                                         // FillSunFoldConstants, so the labels that
                                         // name it can never describe another hour.
    bool  m_BakedGIVisCheck     = true;  // A/B: step-3 fetch weights (bias + backface + Chebyshev) vs plain step-1 trilinear
    int   m_RayDebugProbeIndex  = 0;     // Probe index for the "Dump Probe Ray Debug" tool
    bool  m_PendingRayDebug     = false; // Ray-debug dump request (set by the button)

    // "Probes Near Camera" list — the answer to "which probe index am I standing
    // in?". Filled by BakedGI::FindNearestValidProbes from the live camera
    // position, ascending by distance. A row click sets m_RayDebugProbeIndex and
    // raises m_PendingRayDebug, so one click picks and dumps.
    //
    // The list is CACHED, not rebuilt every frame: rows that re-sort every frame
    // cannot be clicked. It is rebuilt when the camera has moved more than one
    // probe spacing since the build, when the grid revision changes (a re-bake),
    // when the row count changes, or on the Refresh button.
    std::vector<BakedGI::NearestProbe> m_NearestProbes;
    int      m_NearestProbeRows       = 8;     // rows to list (1..32)
    bool     m_NearestProbesValid     = false; // false forces a rebuild
    uint32_t m_NearestProbesGridRev   = 0;     // grid revision the list was built against
    DirectX::XMFLOAT3 m_NearestProbesOrigin = { 0.0f, 0.0f, 0.0f }; // camera position it was built from
    float m_DDGISpacing         = 2.0f;  // DDGI probe grid spacing (m) — used at the next rebuild
    bool  m_PendingDDGISetup    = false; // DDGI grid setup request (auto-set after scene load / Rebuild button)
    int   m_DDGIDebugView       = 0;     // DDGI debug overlay: 0 off, 1 irradiance, 2 probe states, 3 depth
    float m_DDGIDebugScale      = 1.0f;  // DDGI debug gain
    bool  m_IndirectGIEnabled   = true;  // Indirect GI master switch (raster path)
    // Method enum for m_IndirectGIMethod. Declared BEFORE it, because the default
    // initialiser below names one of these constants.
    //
    // The values CHANGED on 2026-10-29, and nothing needed remapping: the method
    // is a plain in-memory member with a default initialiser. It is not a scene
    // field, not an ini key and not a saved setting — a grep for it finds this
    // header and the combo only. "Probe (Baked)" was the old index 0; the binned
    // sun fold it existed for is gone, so it was deleted and the rest moved down.
    //
    // Every "is this the probe method" check goes through ProbeMethodActive(),
    // never a literal.
    static constexpr int kIndirectMethodRestirGI = 0;
    static constexpr int kIndirectMethodDDGI     = 1;
    static constexpr int kIndirectMethodProbe    = 2; // "Probe (Baked Sun)" — the default
    int   m_IndirectGIMethod    = kIndirectMethodProbe;
    bool ProbeMethodActive() const
    {
        return m_IndirectGIMethod == kIndirectMethodProbe;
    }
    // True when the sun column comes from the baked hour states, which is now the
    // only probe method there is. Kept as a separate predicate because the call
    // sites read better against it and because a future probe method that does
    // NOT bake the sun would flip this one alone.
    bool SingleDirectionSun() const
    {
        return m_IndirectGIMethod == kIndirectMethodProbe;
    }
    // The GI sun input: TRUE SOLAR TIME in hours, 0..24, where hour 12 is solar
    // noon. The sun is not a free direction any more — the 17 baked hour states
    // bracket this value and the fold interpolates the pair.
    // Default 12.0, which reproduces the scene's authored sun: state 8 is noon by
    // construction, and its direction equals the authored one to 8.1e-10.
    float m_GIHour = 12.0f;

    // ---- Probe fold dirty flag ----
    //
    // The fold's output is a pure function of the baked tables, the hour and
    // sunE. It is therefore stale only when one of those moves, and this flag
    // says "one of them moved". The fold is 2880 B read and 432 B written for
    // each probe, which is 17.28 GB/s at 100,000 probes and 60 Hz, so running it
    // on a clean frame is pure waste.
    //
    // TRUE at construction, so the first frame folds unconditionally. The four
    // triggers live where the change happens, not here: the hour compare in
    // Application::Render, a completed bake, a method change, and the light
    // editor's own `changed` flag.
    bool  m_ProbeFoldWanted = true;
    // The hour the resident lit table was folded for. -1 is outside 0..24, so the
    // first frame always differs and always folds; it is also the value set after
    // a bake or a method change, for the same reason.
    float m_ProbeFoldHour   = -1.0f;

    // ONE definition of the to-sun direction: the light's travel direction,
    // negated, exactly as BakedGI_Update.hlsl negates it. Four inline copies had
    // accumulated, and the mode-swap check compares DIRECTIONS, so a divergence
    // between two of them would silently disable that check.
    // SunToDirNow returns +Y when the scene has no light, so a caller that cares
    // must ask HasSunLight() as well.
    bool HasSunLight() { return !m_Scene.GetLights().empty(); }
    DirectX::XMFLOAT3 SunToDirNow();
    bool  m_GIIndirectOnly      = false; // Indirect-only scene view: draw just the indirect GI term
    int   m_GIIndirectOnlySource = 0;    // Indirect source split: 0 = full (sky + sun), 1 = sky only, 2 = sun only
    int   m_GIIndirectOnlyAlbedo = 0;    // Indirect-only albedo override: 0 = surface albedo, 1 = all white
    float m_Exposure = -14.0f;
    SDL_Window* m_Window;

    // Core systems
    Renderer m_Renderer;
    Model m_Model;
    Camera m_Camera;
    DirectX::XMMATRIX m_ViewProj;
    DirectX::XMMATRIX m_LastViewMatrix;
    DirectX::XMFLOAT4X4 m_LastViewProj;
    DirectX::XMFLOAT4X4 m_LastViewInverse;
    DirectX::XMFLOAT4 m_LastCameraPos;
    FrameConstants m_FrameConstants;
    FrameConstants m_CullFrameConstants;     // Frozen snapshot used by cull dispatches while m_FreezeCulling is on

    int m_SelectedLightIndex = 0;

    // ImGui
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> m_ImGuiDescriptorHeap;

    // RenderDoc in-application capture — see Sources/Core/RenderDocCapture.h
    RenderDocCapture m_RenderDoc;
    std::string m_RenderDocStatus;            // result of the last capture action

    // Input state for camera
    bool m_RightMouseButtonHeld;
    int m_LastMouseX, m_LastMouseY;

    // Path visualization state
    bool m_PathVizJustClicked = false;
    bool m_PathVizEverCaptured = false;

    // Anti-aliasing / post-processing mode (AA_MODE_NONE, AA_MODE_ACCUMULATION, AA_MODE_TAA)
    int   m_AntiAliasingMode = AA_MODE_TAA;
    float m_TaaUpsamplingFactor = 1.5f;
    bool  m_TaaResetHistory = false;   // Set true when mode/factor changes
    uint32_t m_TaaFrameCounter = 0;   // Monotonically increasing counter for TAA jitter sequence
    bool  m_PendingResolutionChange = false; // Deferred resolution change
    uint32_t m_PendingInternalWidth = 0;
    uint32_t m_PendingInternalHeight = 0;
    uint32_t m_InternalWidth = 0;
    uint32_t m_InternalHeight = 0;
    uint32_t m_OutputWidth = WINDOW_WIDTH;
    uint32_t m_OutputHeight = WINDOW_HEIGHT;

    // Prevent copying
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;
};
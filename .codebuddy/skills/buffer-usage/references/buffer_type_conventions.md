# 🗄️ Buffer Type Conventions

_Reference material for the `buffer-usage` skill — how TortureRed chooses between StructuredBuffer, typed Buffer&lt;T&gt;, ByteAddressBuffer, and ConstantBuffer&lt;T&gt;_

---

## 📋 The four buffer-family types and their C++ backing

| HLSL declaration | C++ creation | SRV/UAV shape (D3D12) | Access pattern |
| --- | --- | --- | --- |
| `StructuredBuffer<T>` / `RWStructuredBuffer<T>` | `CreateStructuredBuffer()` | `Format=UNKNOWN`, `StructureByteStride=sizeof(T)`, `Flags=NONE` | Indexed `buf[i]` of one custom multi-field struct |
| `Buffer<T>` / `RWBuffer<T>` (typed, non-structured) | No dedicated helper yet — manually authored `D3D12_SHADER_RESOURCE_VIEW_DESC` with a real DXGI format | `Format=<real DXGI format>`, `StructureByteStride=0`, `Flags=NONE` | Indexed `buf[i]`; the GPU storage format (uncompressed or packed) drives hardware format conversion on read |
| `ByteAddressBuffer` / `RWByteAddressBuffer` | `CreateBuffer(..., createSRV, createUAV)` | `Format=R32_TYPELESS`, `StructureByteStride=0`, `Flags=RAW` | Manual `.Load()`/`.Store()` at byte offsets — no fixed element type at all |
| `ConstantBuffer<T>` | Root 32-bit constants (`InitAsConstants`) **or** root CBV (`InitAsConstantBufferView`) over an UPLOAD-heap `GPUBuffer` | Not a descriptor-heap SRV at all — root-level binding | Whole-struct read-only |

Grounded in `Sources/Graphics/GraphicsTypes.h:52-53` (declarations) and `GraphicsTypes.cpp:6-121` (implementations)[^1].

---

## 🧭 Readability/writability: three heap types, four usage scenarios

Decide the CPU/GPU access pattern before deciding the element-shape type — it's an orthogonal axis controlled by `D3D12_HEAP_TYPE` and the `createSRV`/`createUAV` arguments. There are three heap types, but the `UPLOAD` heap covers two distinct usage scenarios depending on *how often* the CPU writes:

| Access pattern | Heap type | View(s) created | When to use — example |
| --- | --- | --- | --- |
| **CPU Read/Write + GPU Read** (repeatedly-changing data) | `D3D12_HEAP_TYPE_UPLOAD` | SRV only (`createUAV=false`) | CPU data that keeps changing (e.g. every frame) and must reach the GPU with no copy step — reading per-frame light data, updated via `memcpy` into `cpuPtr` every frame |
| **CPU Read/Write + GPU Read** (one-time initialization → staging) | `D3D12_HEAP_TYPE_UPLOAD` (temporary) | SRV only, short-lived | CPU data written **once** and never changing again, destined for a `DEFAULT`-heap buffer for the best GPU throughput — the UPLOAD buffer is a short-lived staging bridge, discarded right after the GPU-side copy (see [GPU buffer staging](#-gpu-buffer-staging-resourceuploadbatch) below) |
| **GPU Read/Write** | `D3D12_HEAP_TYPE_DEFAULT` | UAV (+ optional SRV) | The GPU produces and consumes the data itself; the CPU never touches it — a compute pass writing its own scratch/result data (reservoirs, hash grids, culling counters) that only another GPU pass reads back |
| **GPU Read/Write + CPU Read** | `D3D12_HEAP_TYPE_READBACK` | — (not currently used) | GPU → CPU readback — e.g. reading a GPU debug/profiling counter back on the CPU (async; not yet implemented anywhere in the project) |

`CreateBuffer(GPUBuffer&, size, heapType, initialState, createSRV, createUAV, debugName)` (`GraphicsTypes.h:52`): passing `D3D12_HEAP_TYPE_UPLOAD` makes `CreateBuffer` automatically `Map()` the resource and cache a persistent `cpuPtr` (`GraphicsTypes.cpp:31-34`) — the CPU can both write *and* read through that pointer (an UPLOAD-heap resource is not write-only), while the GPU only ever reads it. `DeferredLighting::UpdateLightsBuffer()` uses exactly this for the **repeatedly-changing** scenario: `memcpy` into `cpuPtr` every frame (`DeferredLighting.cpp:96`), with no command list or GPU copy involved.

Don't put GPU-only compute scratch data on the UPLOAD heap out of convenience — UPLOAD is CPU-write-optimized, not GPU-throughput-optimized; GPU-only read/write data belongs on `D3D12_HEAP_TYPE_DEFAULT` with a UAV. Conversely, don't leave data that will never change again permanently on the UPLOAD heap either — if it should live on `DEFAULT` for fast GPU access every frame, use the **one-time initialization → staging** scenario instead (next section).

---

## 🚚 GPU buffer staging: `ResourceUploadBatch`

`Sources/Graphics/ResourceUploadBatch.h/.cpp` is TortureRed's shared helper for getting CPU-authored data into a `DEFAULT`-heap buffer: create a temporary `UPLOAD`-heap staging buffer, `memcpy` the data into it, then record a GPU-side `CopyBufferRegion` into the real destination.

**API shape:**

- `ResourceUploadBatch(Renderer*)` — owns its **own** dedicated command allocator + command list (does not reuse a caller-supplied one).
- `Begin()` — resets the allocator/list and clears the staging-buffer list.
- `Upload(GPUBuffer& dest, const void* data, UINT64 size)` — creates a staging buffer via the standard `CreateBuffer()` helper (`UPLOAD` heap, `createSRV=false, createUAV=false`), `memcpy`s into its `cpuPtr`, transitions `dest` to `COPY_DEST`, records a `CopyBufferRegion`, and keeps the staging `GPUBuffer` alive in `m_StagingBuffers` until `End()` — so the caller can queue many `Upload()` calls before releasing anything.
- `Transition(GPUResource&, D3D12_RESOURCE_STATES)` — records a barrier on the batch's own command list (e.g. moving `dest` from `COPY_DEST` to its steady-state, like `GENERIC_READ` or `INDEX_BUFFER`).
- `End()` — closes and executes the command list, then does a full **blocking CPU wait** (its own fence + event) before releasing every staging buffer. Simple and safe — no use-after-free of the staging resource, no deferred-release bookkeeping needed elsewhere — at the cost of stalling the calling thread until the GPU catches up.

**Ground truth — `Model::UploadBuffers()`** (`Model.cpp:1010-1093`). Every global mesh-data buffer created on `D3D12_HEAP_TYPE_DEFAULT` (e.g. `Model.cpp:251,261`: `CreateStructuredBuffer(m_GlobalVertexBuffer, ..., D3D12_HEAP_TYPE_DEFAULT, ...)`) gets its one-time initial content this way, batched into a single `Begin()`/`End()` pair:

```cpp
ResourceUploadBatch batch(renderer);
batch.Begin();
batch.Upload(m_GlobalVertexBuffer, m_GlobalVertices.data(), m_GlobalVertices.size() * sizeof(GLTFVertex));
batch.Transition(m_GlobalVertexBuffer, D3D12_RESOURCE_STATE_GENERIC_READ);
// ... repeated for m_GlobalIndexBuffer, m_MaterialBuffer, m_OpaqueCommandBuffer,
//     m_GlobalPositions/Normals/UVs, m_GlobalMeshlets/Vertices/Triangles/Bounds,
//     m_MeshDataBuffer, m_InstanceBoundsBuffer ...
batch.End();
```

This is the correct bridge for a `DEFAULT`-heap buffer that needs a one-time CPU-authored payload: create it on `DEFAULT` for GPU throughput, then push the initial data through one `ResourceUploadBatch` pass rather than leaving the buffer permanently on `UPLOAD` heap (which would make every frame's GPU reads slower for no benefit, since the data never changes after load).

---

## 🧮 Choosing the element-shape type

Once the heap/access pattern is decided, pick the HLSL type based on the **shape of one element**, not on habit. This is a strict decision order:

| Data shape | Correct type | Why |
| --- | --- | --- |
| Array of one **custom struct** with multiple named fields | `StructuredBuffer<T>` / `RWStructuredBuffer<T>` (default) | Only view that reflects field names in RenderDoc/PIX |
| Array where each element **maps to one real DXGI hardware format** — whether an uncompressed 1:1 format (`R32_UINT`, `R32_FLOAT`) or a packed/compressed one (`R8G8_SNORM`, `R16_FLOAT`, ...) | Typed `Buffer<T>` / `RWBuffer<T>` with that DXGI format | A typed buffer *is* a GPU hardware format choice — that's the only thing distinguishing it from `StructuredBuffer`/`ByteAddressBuffer`, which never carry a format and only reinterpret raw bytes. The deciding question is always "what DXGI format does this GPU memory represent," not "is the data simple or fancy" |
| **Multiple different types/counters** packed at manually-managed byte offsets in one resource — no single element format/stride describes the whole buffer | `ByteAddressBuffer` / `RWByteAddressBuffer` | The only view family with no fixed element type/format at all; use only when the data genuinely cannot be described by one struct or one DXGI format |

`ByteAddressBuffer` is the **narrowest, last-resort** case — reach for it only when a single resource must hold heterogeneous data that no fixed-element view (structured or typed) can describe. It is not a substitute for "I don't want to think about the type" — that has a better-reflecting, more specific answer above.

### Case A — StructuredBuffer (default): custom multi-field structs

The overwhelming majority of buffers in the project — `m_GlyphData`, `m_IndirectArgs`, `m_RtxdiReservoirBuffer[2]`, `g_DrawNodeBuffer`, `g_Materials`, `g_GlobalVertices`, `g_GlobalIndices` — are `StructuredBuffer`/`RWStructuredBuffer` of a custom struct. Because the SRV/UAV carries `StructureByteStride` and the HLSL declares the exact struct `T`, RenderDoc/PIX show named, typed columns (e.g. `DebugGlyph.MinUV`, `DebugGlyph.AdvanceX`) instead of a flat hex dump.

### Case B — typed Buffer&lt;T&gt;: driven by the GPU storage format, not the CPU type

A typed `Buffer<T>` is defined by an explicit DXGI hardware format on its SRV/UAV — that format *is* what makes it "typed" at all, and it is a deliberate GPU-side storage decision, not just a mirror of whatever scalar type the HLSL happens to declare. Every example below is the same mechanism; the only variable is *which* format matches the data:

**Example 1 — `R32_UINT`, an uncompressed 1:1 format.** `Sources/Rendering/DeferredLighting.cpp:106` creates `m_LightLUTBuffer` as a flat array of 256 `uint`s (`CommonTracing.hlsl:14` reads it as `g_LightLUT`). There is no heterogeneous layout here, so per the decision table this is a typed buffer with format `DXGI_FORMAT_R32_UINT`:

```hlsl
Buffer<uint> g_LightLUT : register(t1, space2);
// ...
uint lightIdx = g_LightLUT[lutIndex];   // indexed read, not a byte offset
```

> 📌 **Current code status:** the actual codebase declares this as `ByteAddressBuffer g_LightLUT` with `.Load(lutIndex * 4)` byte addressing, created via `CreateBuffer()`'s raw path. Per this guideline that's the wrong category — a flat `uint[256]` has exactly one GPU storage format (`R32_UINT`), so it doesn't need raw addressing, and it isn't a struct either (a "struct with one `uint` field" is not what a plain scalar array is, so `StructuredBuffer<uint>` is also wrong here). The corrected form is a typed `Buffer<uint>` (SRV with `Format=DXGI_FORMAT_R32_UINT`, `StructureByteStride=0`, `Flags=NONE`, manually authored the same way as Example 2 below). This is a known, flagged cleanup item; fixing the live shader/C++ code is a separate follow-up from this documentation.

**Example 2 — `R8G8_SNORM`, a packed/compressed format.** `Sources/Rendering/PathTracing.cpp:86-109`: RTXDI's `rtxdi::FillNeighborOffsetBuffer()` (vendored NVIDIA RTXDI SDK) fills a 2-byte-per-element signed-normalized disc-offset table — the GPU storage format for this data genuinely *is* `R8G8_SNORM`, not `R32G32_FLOAT`, because that is how RTXDI chose to pack it. The SRV is manually authored with `Format = DXGI_FORMAT_R8G8_SNORM` (`PathTracing.cpp:101`), and the HLSL side declares it as a typed buffer:

```hlsl
// RestirGI_RTXDI_Temporal.hlsl:6, RestirGI_RTXDI_Spatial.hlsl:6
Buffer<float2> g_NeighborOffsets : register(t5, space1);
```

Reading `g_NeighborOffsets[i]` returns a `float2` in `[-1, 1]`; the GPU's fixed-function format-conversion hardware decodes the 2-byte SNORM8x2 storage into a full `float2` on read. A `StructuredBuffer<float2>` could not do this — it has no concept of a hardware format at all, so it would force the underlying storage to already be 8-byte float2s (4× the memory), which contradicts the format RTXDI's SDK actually produces.

Both examples pick a DXGI format for the *actual* GPU-resident bytes — one happens to be an uncompressed passthrough format, the other a packed one. That is the entire distinction; neither needs `ByteAddressBuffer`, and neither is "simpler" or "fancier" than the other.

### Case C — ByteAddressBuffer: heterogeneous mixed-layout buffers only

A `StructuredBuffer<T>`/typed `Buffer<T>` view is locked to **one fixed-stride element type** for the entire resource. If a single buffer must pack *different* things at different byte ranges — atomic counters followed by two differently-typed instance arrays — no fixed-element view can express that.

**Ground truth — `DebugTextRenderer`'s `m_RenderData` buffer** (`Sources/Rendering/DebugTextRenderer.h:48`, comment: `// ByteAddress (RAW): counters + instances`). One buffer packs, at fixed byte offsets defined in `Sources/Shared/SharedTypes.h:404-412`:

```cpp
#define DEBUG_TEXT_MAX_CHARS 8192
#define DEBUG_TEXT_MAX_LINES 32768
#define DEBUG_TEXT_COUNTER_OFFSET    0   // uint: text-char count (atomic)
#define DEBUG_LINE_COUNTER_OFFSET    4   // uint: line count (atomic)
#define DEBUG_TEXT_COUNTERS_SIZE     16  // 2 counters + padding
#define DEBUG_TEXT_INSTANCES_OFFSET  DEBUG_TEXT_COUNTERS_SIZE
#define DEBUG_LINE_INSTANCES_OFFSET  (DEBUG_TEXT_INSTANCES_OFFSET + DEBUG_TEXT_MAX_CHARS * 32)
```

Producers append via `InterlockedAdd` on the raw buffer (`DebugTextRender.hlsli:46,70`) then `Store` instance data at `DEBUG_TEXT_INSTANCES_OFFSET + slot * 32` / `DEBUG_LINE_INSTANCES_OFFSET + slot * 32`. **`InterlockedAdd` is not itself a reason to choose raw** — `RWStructuredBuffer<uint>` and `RWBuffer<uint>` both support atomics too — the raw choice here is driven entirely by the buffer holding two counters *and* two different instance-array types in one resource, which is the one genuine reason to fall back to `ByteAddressBuffer`.

> ⚠️ **Pitfall — `Store`/`Load<T>` struct decomposition mismatch.** `DebugTextRender.hlsli:50-54` documents a real bug class: `ByteAddressBuffer::Store` of a struct containing a `float4` can decompose members at different byte offsets than `Load<T>` reassembles them, particularly once the compiler inserts 16-byte alignment padding for the `float4`. `DebugAddCharacter` therefore stores every field with an **explicit dword-by-dword `Store(base + N, ...)`** rather than `Store<DebugCharInstance>(base, inst)`, to guarantee producer and consumer agree on layout byte-for-byte. Follow this pattern for any new packed-struct write into a raw buffer — do not trust `Store<T>`/`Load<T>` symmetry for structs containing `float4`/`float3` members.

---

## 🔧 `ConstantBuffer<T>`: root constants vs. root CBV

Two backing mechanisms exist for `ConstantBuffer<T>` in the main root signature (`Sources/Rendering/Renderer.cpp:648-670`):

| Mechanism | Root signature call | Resource? | Used for |
| --- | --- | --- | --- |
| Root 32-bit constants | `InitAsConstants(numDwords, shaderRegister, space)` + `SetXRoot32BitConstants(rootParamIndex, ...)` | None — data lives inline in the command list | Small, per-draw/per-dispatch parameter blocks: `BindlessIndices`, `RasterParams`, `DebugTextRenderParams` (all share root param 12 / `b1`), `IrCacheBindlessIndices` (root param 13 / `b2`) |
| Root CBV | `InitAsConstantBufferView(shaderRegister)` + `SetXRootConstantBufferView(rootParamIndex, gpuAddress)` | Yes — an UPLOAD-heap `GPUBuffer` with a persistent `cpuPtr` | Larger/shared per-frame data: `FrameConstants` (root param 0 / `b0`); also used for smaller per-technique constants that need a stable GPU address across a pass (e.g. `TwoPassCullConstants`, `HZBConstants`) |

> ⚠️ **Pitfall — shared root-constants slot sizing.** Root param 12 (`b1`) is reused, unaligned, by several unrelated structs across different passes (`Renderer.cpp:660-668`): `RasterParams`, `BindlessIndices`, `DebugTextRenderParams`. Its declared `Num32BitValues` must cover the **largest** of them:
> ```cpp
> constexpr size_t kParam12MaxA = std::max(sizeof(DebugTextRenderParams), std::max(sizeof(RasterParams), sizeof(BindlessIndices)));
> rootParameters[12].InitAsConstants(static_cast<UINT>(kParam12MaxA / 4), 1, 0);
> ```
> If you add a **new** struct to this shared slot, or grow an existing one, you must update `kParam12MaxA`'s `std::max(...)` chain — otherwise `SetComputeRoot32BitConstants(12, ...)`/`SetGraphicsRoot32BitConstants(12, ...)` calls from the now-larger struct silently write past the declared root-signature capacity.

---

## ⚠️ Common pitfalls (summary)

- **Reaching for `ByteAddressBuffer` for a flat array of one scalar type.** Don't — see [Case B](#case-b--typed-buffert-driven-by-the-gpu-storage-format-not-the-cpu-type); use a typed `Buffer<T>` with the matching DXGI format instead. `ByteAddressBuffer` is only for genuinely heterogeneous mixed-layout buffers ([Case C](#case-c--byteaddressbuffer-heterogeneous-mixed-layout-buffers-only)).
- **Trusting `Store<T>`/`Load<T>` symmetry for structs with `float4`/`float3` members in a raw buffer.** Use explicit dword-by-dword `Store`/`Load` instead (see Case C).
- **Assuming `ExecuteIndirect` requires a raw buffer for its args.** It doesn't — `ExecuteIndirect` reads directly from the resource's GPU memory at a byte offset, bypassing SRV/UAV views entirely. `DebugTextRenderer`'s indirect-args buffer is a plain `StructuredBuffer` (`CreateStructuredBuffer(m_IndirectArgs, 16, 2, ...)`), not raw.
- **Growing a struct that shares a root-constants slot** (`b1`/root param 12) without updating the `std::max(...)` sizing chain in `Renderer.cpp`.
- **Using an UPLOAD-heap buffer for GPU-only compute scratch data.** UPLOAD is optimized for CPU writes; GPU-only read/write data belongs on `D3D12_HEAP_TYPE_DEFAULT` with a UAV.
- **Introducing a new non-bindless root-SRV-by-GPU-address binding.** A few fixed-slot buffers (model material/vertex/index buffers, the lights buffer, the light LUT, the TLAS) are still bound via `SetGraphicsRootShaderResourceView(rootParamIndex, gpuAddress)` (`DeferredLighting.cpp:207-210`) instead of the bindless `ResourceDescriptorHeap[idx]` pattern. **This is legacy and will be replaced** — the project is moving toward fully bindless resource access. Do not copy this pattern for new buffers; always create an SRV/UAV descriptor-heap slot (`CreateBuffer`/`CreateStructuredBuffer`) and index it bindlessly via a constant, even for buffers that happen to be bound every draw today.

---

## 🔗 References

[^1]: Microsoft. "Buffer Views (Raw, Structured, and Typed) — D3D12 Programming Guide." _Direct3D 12 Graphics_. https://learn.microsoft.com/en-us/windows/win32/direct3d12/uav-creation

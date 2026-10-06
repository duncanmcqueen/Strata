#pragma once
// include/strata/sycl_compat/cuda_runtime.h - the SYCL backend's CUDA runtime shim.
//
// Included only by STRATA_ENABLE_SYCL builds, force-included into every C++ translation
// unit (cmake/sycl_backend.cmake), exactly as the HIP shim is.  Where the HIP shim is a
// table of macro renames onto a CUDA-shaped runtime, SYCL has no drop-in runtime, so
// every name here is a DECLARED host function, implemented by the strata_sycl_runtime
// library (src/sycl_runtime/) on top of SYCL USM, in-order queues and the
// sycl_ext_oneapi_graph extension.  Handles are opaque pointers to small structs:
//   cudaStream_t     -> in-order sycl::queue plus capture state
//   cudaEvent_t      -> sycl::event plus record/timing state
//   cudaGraphExec_t  -> a finalized (executable) SYCL command graph
// Nothing but the subset Strata actually calls is declared (the list the HIP shim
// enumerates); when a caller needs more, add it here AND implement it.
//
// Keep this header free of SYCL includes: it lands in every translation unit,
// including pure host code.  Kernel sources that need the sycl::queue behind a stream
// include "strata/sycl_runtime/queue_bridge.hpp" instead.

#include <cstddef>
#include <cstdint>

// The CUDA runtime version the shim's cudaRuntimeGetVersion/cudaDriverGetVersion report (a shape, not a toolkit):
// a caller comparing its headers against the loaded runtime sees them agree.
#define CUDART_VERSION 12080

// ---- errors -----------------------------------------------------------------
// The values mirror CUDA's where the meaning matches; SYCL reports errors by
// exception, so these are produced by the shim (and its queue async handler).
enum cudaError_t {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorNoDevice = 100,
    cudaErrorUnknown = 30,
    cudaErrorNotReady = 34,
    cudaErrorPeerAccessAlreadyEnabled = 104,
    cudaErrorStreamCaptureUnsupported = 901,
    cudaErrorStreamCaptureInvalidated = 902,
};
cudaError_t cudaGetLastError();
cudaError_t cudaPeekAtLastError();
const char* cudaGetErrorString(cudaError_t error);

// ---- opaque handles ----------------------------------------------------------
typedef struct strata_cuda_stream* cudaStream_t;
typedef struct strata_cuda_event* cudaEvent_t;
typedef struct strata_cuda_graph* cudaGraph_t;
typedef struct strata_cuda_graph_exec* cudaGraphExec_t;
typedef struct strata_cuda_graph_node* cudaGraphNode_t;

// ---- devices ------------------------------------------------------------------
struct cudaDeviceProp {
    char name[256];
    size_t totalGlobalMem;
    int l2CacheSize;  // SYCL global-memory cache size, used as a benchmark sizing estimate.
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;                     // KHz, as CUDA reports it
    size_t totalConstMem;
    int major;                         // compute capability sentinel: see cudaDeviceGetAttribute
    int minor;
    int multiProcessorCount;           // Xe-core count (slices x subslices per slice)
    int integrated;
    int cooperativeLaunch;
    int maxThreadsPerMultiProcessor;
    size_t sharedMemPerMultiprocessor;
    int pciBusID;
    int pciDeviceID;
};

enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrClockRate = 13,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrWarpSize = 10,
};

cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int* device);
// The scheduling flags have no Level Zero counterpart (host waits poll the runtime either way); they are accepted
// so a caller's spin-scheduling request is a no-op, as cudaSetDeviceFlags would be on a context that exists.
enum { cudaDeviceScheduleAuto = 0x00, cudaDeviceScheduleSpin = 0x01, cudaDeviceScheduleYield = 0x02,
       cudaDeviceScheduleBlockingSync = 0x04, cudaDeviceMapHost = 0x08 };
cudaError_t cudaInitDevice(int device, int deviceFlags, int flags);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device);
cudaError_t cudaDeviceSynchronize();
cudaError_t cudaDeviceCanAccessPeer(int* canAccess, int device, int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);
cudaError_t cudaDriverGetVersion(int* driverVersion);

// ---- memory -------------------------------------------------------------------
enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

// Host allocation flags, same values as CUDA.
enum {
    cudaHostAllocDefault = 0x00,
    cudaHostAllocPortable = 0x01,
    cudaHostAllocMapped = 0x02,
    cudaHostAllocWriteCombined = 0x04,
};
enum {
    cudaHostRegisterDefault = 0x00,
    cudaHostRegisterPortable = 0x01,
    cudaHostRegisterMapped = 0x02,
    cudaHostRegisterReadOnly = 0x04,
};

cudaError_t cudaMalloc(void** pointer, size_t bytes);
cudaError_t cudaFree(void* pointer);
cudaError_t cudaMallocHost(void** pointer, size_t bytes);
cudaError_t cudaHostAlloc(void** pointer, size_t bytes, unsigned int flags);
cudaError_t cudaFreeHost(void* pointer);
cudaError_t cudaHostGetDevicePointer(void** devicePointer, void* hostPointer, unsigned int flags);
// SYCL has no host-memory registration: the arena is allocated with malloc_host in the
// SYCL port of pinned.cu instead (PageBacking::PinnedBySycl).  These fail loudly so a
// missed call site cannot silently run unpinned.
cudaError_t cudaHostRegister(void* pointer, size_t bytes, unsigned int flags);
cudaError_t cudaHostUnregister(void* pointer);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream = nullptr);
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dstPitch, const void* src, size_t srcPitch, size_t width,
                              size_t height, cudaMemcpyKind kind, cudaStream_t stream);
// Peer (cross-card) copies are out of scope for v1; this returns cudaErrorInvalidValue.
cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t bytes,
                                cudaStream_t stream);
cudaError_t cudaMemset(void* pointer, int value, size_t bytes);
cudaError_t cudaMemsetAsync(void* pointer, int value, size_t bytes, cudaStream_t stream = nullptr);
cudaError_t cudaMemGetInfo(size_t* freeBytes, size_t* totalBytes);

// The CUDA headers type-pun through templates so cudaMalloc(&typed_ptr, ...) compiles;
// keep that convenience here or every caller would need a cast.
template <typename T>
cudaError_t cudaMalloc(T** pointer, size_t bytes) {
    return cudaMalloc(reinterpret_cast<void**>(pointer), bytes);
}
template <typename T>
cudaError_t cudaMallocHost(T** pointer, size_t bytes) {
    return cudaMallocHost(reinterpret_cast<void**>(pointer), bytes);
}
template <typename T>
cudaError_t cudaHostAlloc(T** pointer, size_t bytes, unsigned int flags) {
    return cudaHostAlloc(reinterpret_cast<void**>(pointer), bytes, flags);
}
template <typename T>
cudaError_t cudaHostGetDevicePointer(T** devicePointer, T* hostPointer, unsigned int flags) {
    return cudaHostGetDevicePointer(reinterpret_cast<void**>(devicePointer), reinterpret_cast<void*>(hostPointer),
                                    flags);
}

// ---- streams -------------------------------------------------------------------
enum { cudaStreamDefault = 0x00, cudaStreamNonBlocking = 0x01 };

cudaError_t cudaStreamCreate(cudaStream_t* stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags = 0);

typedef void (*cudaHostFn_t)(void* userData);
// Recorded as a host_task on the stream's queue; REFUSED during graph capture
// (cudaErrorStreamCaptureUnsupported) so the caller's non-captured fallback runs:
// DPC++ does not record host tasks into command graphs.
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData);

// ---- events ---------------------------------------------------------------------
enum { cudaEventDefault = 0x00, cudaEventBlockingSync = 0x01, cudaEventDisableTiming = 0x02 };

cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);

// ---- graphs -----------------------------------------------------------------------
// Implemented on sycl_ext_oneapi_graph: BeginCapture starts recording the stream's
// queue, EndCapture stops and hands back the modifiable graph, Instantiate finalizes
// it, Launch submits the executable graph to the stream's queue, Upload is a no-op.
// Anything the installed DPC++ release refuses to record fails BeginCapture or the
// offending call with cudaErrorStreamCaptureUnsupported so Strata's existing
// non-captured fallbacks take over.  The three graph rules of
// include/strata/core/graph.hpp hold unchanged: kernel arguments are frozen at record
// time, every recorded buffer is allocated before recording and never moved, and
// waits poll the runtime (cudaEventQuery), never memory.
enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
};
enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2,
};
enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeCount = 6,
};
struct dim3 {
    unsigned int x, y, z;
};
struct cudaKernelNodeParams {
    dim3 blockDim;
    unsigned int dynamicSMemBytes;
    void* func;
    dim3 gridDim;
    void** kernelParams;
    void** extra;
    unsigned int sharedMemBytes;
};

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* status);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long flags);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, cudaGraphNode_t* errorNode, char* logBuffer,
                                 size_t bufferSize);
cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream = nullptr);
cudaError_t cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t stream);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec);
// Node count/handles and supported command types map to SYCL graph inspection.
// Kernel launch-parameter inspection remains unsupported.
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* type);
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, cudaKernelNodeParams* params);

// ---- function attributes / occupancy ----------------------------------------------
enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9,
};
struct cudaFuncAttributes {
    size_t sharedSizeBytes;
    size_t constSizeBytes;
    size_t localSizeBytes;
    int maxThreadsPerBlock;
    int numRegs;
    int ptxVersion;
    int binaryVersion;
    int cacheModeCA;
    int maxDynamicSharedSizeBytes;
    int preferredShmemCarveout;
};
// SYCL kernel launches size local memory at submit time, so there is no opt-in to set:
// the request is validated against the device's local memory size and either accepted
// (a no-op) or refused loudly, never silently dropped.
cudaError_t cudaFuncSetAttribute(const void* func, cudaFuncAttribute attribute, int value);
// A SYCL queue cannot map a host function pointer back to a registered kernel, so
// attributes and occupancy queries are unavailable: they fail loudly.  The prefill
// callers are revisited with their kernels.
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attributes, const void* func);
cudaError_t cudaFuncGetName(const char** name, const void* func);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize,
                                                          size_t dynamicSMemSize);
template <typename Kernel>
cudaError_t cudaFuncSetAttribute(Kernel kernel, cudaFuncAttribute attribute, int value) {
    return cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
template <typename Kernel>
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attributes, Kernel kernel) {
    return cudaFuncGetAttributes(attributes, reinterpret_cast<const void*>(kernel));
}
template <typename Kernel>
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, Kernel kernel, int blockSize,
                                                          size_t dynamicSMemSize) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, reinterpret_cast<const void*>(kernel), blockSize,
                                                         dynamicSMemSize);
}

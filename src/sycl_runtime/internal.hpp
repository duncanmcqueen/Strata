#pragma once
// src/sycl_runtime/internal.hpp - shared internals of the strata_sycl_runtime library.
// NOT installed under include/: only the four runtime translation units include this.
// Kernel sources use the public bridge ("strata/sycl_runtime/queue_bridge.hpp").

#include <sycl/sycl.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace syclexp = sycl::ext::oneapi::experimental;

struct strata_cuda_stream {
    sycl::queue queue;  // in-order, profiling-enabled
    int device = 0;
    bool capturing = false;
    bool blocking = true;  // created without cudaStreamNonBlocking: ordered with the legacy default stream
    struct strata_cuda_graph* graph = nullptr;  // the graph recording this queue
};

struct strata_cuda_event {
    sycl::event event;
    bool recorded = false;
    bool timing = true;  // events created with cudaEventDisableTiming report elapsed as 0
    int device = 0;
};

struct strata_cuda_graph_node {
    syclexp::node node;
};

struct strata_cuda_graph {
    std::unique_ptr<syclexp::command_graph<syclexp::graph_state::modifiable>> graph;
    std::vector<strata_cuda_stream*> recording;  // queues handed to begin_recording
    int device = 0;
    std::vector<std::unique_ptr<strata_cuda_graph_node>> nodes;
};

struct strata_cuda_graph_exec {
    std::unique_ptr<syclexp::command_graph<syclexp::graph_state::executable>> graph;
    int device = 0;
};

namespace strata::sycl_runtime {

// Device registry: the Level Zero GPUs of ONEAPI_DEVICE_SELECTOR that support
// 32-wide sub-groups (Strata's kernels assume 32-lane warps; a GPU without
// sub-group 32 is excluded at enumeration, so the backend refuses to start on it).
// One shared context spans all of them so USM pointers are valid on every queue.
struct Registry {
    std::vector<sycl::device> devices;
    sycl::context context;
    std::vector<std::unique_ptr<strata_cuda_stream>> default_streams;  // per device, stream == nullptr
    std::vector<sycl::queue*> queues;  // every live queue, so device synchronize sees them all
    std::vector<strata_cuda_stream*> blocking_streams;  // live created streams without cudaStreamNonBlocking
};
Registry& registry();
std::mutex& queues_mutex();

// Queues live inside stream structs (stable addresses); registered at creation so
// cudaDeviceSynchronize waits every stream, not just the default one.
void register_queue(sycl::queue* q);
void unregister_queue(sycl::queue* q);
void register_blocking_stream(strata_cuda_stream* s);
// STRATA_SYCL_TRACE: a host-side note in the hang locator's log (graph capture and launch)
void trace_event(const std::string& what);
/// How far into its device allocation (cudaMalloc) the byte range [p, p + bytes) ends; 0 when p is not in one.
size_t device_offset_end(const void* p, size_t bytes);
/// STRATA_SYCL_QSTATS=1: count submissions per stream and kind, with the first caller; printed at exit
void note_submit(const void* stream, const char* kind, void* caller);
bool trace_on();
void trace_end_marker(sycl::queue& q);
void unregister_blocking_stream(strata_cuda_stream* s);

// CUDA's legacy default stream: a synchronous cudaMemcpy/cudaMemset on it starts only after all work already
// submitted to the device's BLOCKING streams.  Waits for those queues (streams being captured are skipped: CUDA
// refuses that implicit synchronization during capture, and waiting on a recording queue has nothing to wait for).
// Work submitted later is ordered after the synchronous call because the call returns only once it completes.
void wait_blocking_streams(int device);
// a synchronization point's result: the recorded asynchronous error, else success (runtime.cpp)
cudaError_t sync_result();

int current_device();                                    // the thread's cudaSetDevice choice
cudaError_t set_current_device(int device);
strata_cuda_stream* default_stream(int device);          // the legacy default stream of a device
strata_cuda_stream* stream_or_default(cudaStream_t stream);  // nullptr -> default of current device

// Error plumbing.  SYCL reports by exception: API calls catch and translate, and every
// queue's async handler funnels into the same slot that cudaGetLastError drains.
cudaError_t set_last_error(cudaError_t error);
cudaError_t fail(cudaError_t error);  // set + return, the one-liner at every error exit
cudaError_t from_exception(const sycl::exception& e);
void note_async_error(const char* what);

// Allocation accounting for the cudaMemGetInfo fallback (when sysman free-memory is
// unavailable): total minus what this process currently holds.
extern std::atomic<int64_t> g_allocated_bytes;

}  // namespace strata::sycl_runtime

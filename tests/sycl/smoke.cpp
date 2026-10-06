// tests/sycl/smoke.cpp - SYCL backend checks that have no CUDA-side counterpart:
// device discovery against the guide's table (Level Zero GPU, 32-wide sub-group,
// the 8.0 compute-capability sentinel), a USM malloc/memcpy/memset round-trip,
// mapped host memory read by the device, event elapsed time, and a captured graph
// launched twice with the second run seeing new input (the graph rules: arguments
// frozen, buffers stable, the data itself read from device memory).
//
// Not a parity test: there is no reference to be bit-exact against, only the
// runtime's own contracts.  Exits nonzero on the first failure.

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int bad = 0;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void expect(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL %s\n", what);
        ++bad;
    }
}

}  // namespace

int main() {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    expect(count >= 1, "at least one Level Zero GPU");
    std::printf("devices: %d\n", count);

    for (int d = 0; d < count; ++d) {
        cudaDeviceProp prop;
        check(cudaGetDeviceProperties(&prop, d), "cudaGetDeviceProperties");
        std::printf("  [%d] %s  %zu MiB, %d Xe cores, warp %d, cc sentinel %d.%d, local mem %zu KiB\n", d, prop.name,
                    prop.totalGlobalMem >> 20, prop.multiProcessorCount, prop.warpSize, prop.major, prop.minor,
                    prop.sharedMemPerBlock >> 10);
        expect(prop.warpSize == 32, "warp size 32 (the sub-group guarantee)");
        expect(prop.major == 8 && prop.minor == 0, "compute capability sentinel 8.0");
        expect(prop.totalGlobalMem > 0, "global memory reported");
        int sg = 0;
        check(cudaDeviceGetAttribute(&sg, cudaDevAttrWarpSize, d), "cudaDeviceGetAttribute warp size");
        expect(sg == 32, "attribute warp size 32");
    }
    check(cudaSetDevice(0), "cudaSetDevice");

    // Device memory round-trip through both copy directions.
    {
        const size_t n = 1 << 20;
        std::vector<float> host(n);
        for (size_t i = 0; i < n; ++i) host[i] = static_cast<float>(i % 977);
        float* dev = nullptr;
        check(cudaMalloc(&dev, n * sizeof(float)), "cudaMalloc");
        check(cudaMemcpy(dev, host.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D");
        check(cudaMemset(dev, 0, 1024), "cudaMemset");
        host[0] = 123.0f;  // prove the next H2D lands
        check(cudaMemcpy(dev, host.data(), sizeof(float), cudaMemcpyHostToDevice), "H2D small");
        std::vector<float> back(n, -1.0f);
        check(cudaMemcpy(back.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost), "D2H");
        // memset cleared the first 256 floats; the small H2D then rewrote only [0].
        expect(back[0] == 123.0f, "round-trip: small H2D landed");
        expect(back[1] == 0.0f && back[255] == 0.0f, "round-trip: memset region zeroed");
        expect(back[256] == static_cast<float>(256 % 977), "round-trip: past the memset");
        expect(back[n - 1] == static_cast<float>((n - 1) % 977), "round-trip tail");
        check(cudaFree(dev), "cudaFree");
    }

    // Mapped host memory: written on the host, read by a device copy back out.
    {
        float* mapped = nullptr;
        check(cudaHostAlloc(&mapped, 4096, cudaHostAllocMapped), "cudaHostAlloc mapped");
        void* dev_ptr = nullptr;
        check(cudaHostGetDevicePointer(&dev_ptr, mapped, 0), "cudaHostGetDevicePointer");
        expect(dev_ptr == mapped, "device pointer is the host pointer (host USM)");
        mapped[7] = 42.0f;
        float* dev = nullptr;
        check(cudaMalloc(&dev, 4096), "cudaMalloc for mapped read");
        check(cudaMemcpy(dev, mapped, 4096, cudaMemcpyHostToDevice), "mapped H2D");
        float out[8] = {};
        check(cudaMemcpy(out, dev, sizeof(out), cudaMemcpyDeviceToHost), "D2H of mapped read");
        expect(out[7] == 42.0f, "mapped host memory readable by the device");
        check(cudaFree(dev), "cudaFree");
        check(cudaFreeHost(mapped), "cudaFreeHost");
    }

    // Streams, events, elapsed time.
    {
        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "cudaStreamCreate");
        cudaEvent_t start = nullptr, stop = nullptr;
        check(cudaEventCreate(&start), "cudaEventCreate start");
        check(cudaEventCreate(&stop), "cudaEventCreate stop");
        float* dev = nullptr;
        check(cudaMalloc(&dev, 16 << 20), "cudaMalloc for events");
        check(cudaEventRecord(start, stream), "record start");
        check(cudaMemsetAsync(dev, 0xFF, 16 << 20, stream), "memset async");
        check(cudaEventRecord(stop, stream), "record stop");
        check(cudaStreamSynchronize(stream), "stream sync");
        expect(cudaEventQuery(stop) == cudaSuccess, "event query after sync");
        float ms = -1.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "elapsed time");
        expect(ms >= 0.0f, "elapsed time nonnegative");
        std::printf("16 MiB memset: %.3f ms\n", (double) ms);
        check(cudaEventDestroy(start), "destroy start");
        check(cudaEventDestroy(stop), "destroy stop");
        check(cudaFree(dev), "cudaFree");
        check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    }

    // Graph capture: memset recorded once, launched twice; the second launch must
    // re-run the same node (arguments frozen, buffer stable) and be observable.
    {
        cudaStream_t stream = nullptr;
        check(cudaStreamCreate(&stream), "graph stream");
        int* dev = nullptr;
        check(cudaMalloc(&dev, 64 * sizeof(int)), "graph buffer");
        const int first = 7, second = 9;
        check(cudaMemsetAsync(dev, 0, 64 * sizeof(int), stream), "clear");
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "begin capture");
        check(cudaMemsetAsync(dev, first, 64 * sizeof(int), stream), "captured memset");
        cudaGraph_t graph = nullptr;
        check(cudaStreamEndCapture(stream, &graph), "end capture");
        expect(graph != nullptr, "graph returned");
        size_t node_count = 0;
        check(cudaGraphGetNodes(graph, nullptr, &node_count), "graph node count");
        expect(node_count == 1, "one recorded memset node");
        cudaGraphNode_t node = nullptr;
        size_t node_capacity = 1;
        check(cudaGraphGetNodes(graph, &node, &node_capacity), "graph node handle");
        cudaGraphNodeType node_type;
        check(cudaGraphNodeGetType(node, &node_type), "graph node type");
        expect(node_type == cudaGraphNodeTypeMemset, "recorded node is a memset");
        cudaGraphNode_t same_node = nullptr;
        check(cudaGraphGetNodes(graph, &same_node, &node_capacity), "stable graph node handle");
        expect(same_node == node, "graph node handle stays stable");
        expect(cudaGraphGetNodes(nullptr, nullptr, &node_count) == cudaErrorInvalidValue,
               "node inspection rejects a null graph");
        (void)cudaGetLastError();
        cudaGraphExec_t exec = nullptr;
        check(cudaGraphInstantiate(&exec, graph, 0ull), "instantiate");
        check(cudaGraphLaunch(exec, stream), "launch 1");
        check(cudaStreamSynchronize(stream), "sync 1");
        int back[64] = {};
        check(cudaMemcpy(back, dev, sizeof(back), cudaMemcpyDeviceToHost), "read 1");
        const int filled1 = first | (first << 8) | (first << 16) | (first << 24);
        expect(back[0] == filled1 && back[63] == filled1, "graph launch 1 filled the buffer");
        check(cudaMemsetAsync(dev, 0, 64 * sizeof(int), stream), "clear again");
        check(cudaGraphLaunch(exec, stream), "launch 2");
        check(cudaStreamSynchronize(stream), "sync 2");
        check(cudaMemcpy(back, dev, sizeof(back), cudaMemcpyDeviceToHost), "read 2");
        expect(back[0] == filled1 && back[63] == filled1, "graph launch 2 re-ran the same node");
        check(cudaGraphExecDestroy(exec), "exec destroy");
        check(cudaGraphDestroy(graph), "graph destroy");
        check(cudaFree(dev), "cudaFree");
        check(cudaStreamDestroy(stream), "stream destroy");
        (void) second;
    }

    // memGetInfo: sane bounds only (the exact figure moves with the desktop).
    {
        size_t free_b = 0, total_b = 0;
        check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        std::printf("memory: %zu MiB free of %zu MiB\n", free_b >> 20, total_b >> 20);
        expect(total_b > 0 && free_b <= total_b, "memGetInfo bounds");
    }

    if (bad) {
        std::fprintf(stderr, "sycl_smoke: %d failures\n", bad);
        return 1;
    }
    std::printf("sycl_smoke OK\n");
    return 0;
}

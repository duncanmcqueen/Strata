// src/sycl_runtime/graph.cpp - CUDA stream capture and graphs on sycl_ext_oneapi_graph.
//
// Mapping: BeginCapture starts recording the stream's in-order queue into a fresh
// modifiable command graph, EndCapture stops recording and hands the graph out,
// Instantiate finalizes it, Launch submits the executable graph to the stream's
// queue, Upload is a no-op (Level Zero builds the executable at finalize).
//
// What CUDA would record beyond the captured stream - work other streams submit
// while capture is active - SYCL does NOT: only the recording queue's submissions
// become nodes.  Strata captures single-stream bodies (the layer graphs), so this
// matches; a cross-stream body would need its events wired with cudaStreamWaitEvent,
// which records as a graph edge through the queue's barrier.
//
// Anything the installed DPC++ release refuses to record fails with
// cudaErrorStreamCaptureUnsupported so Strata's existing non-captured fallbacks take
// over (the porting guide's graph rule).  Host functions are refused at
// cudaLaunchHostFunc (runtime.cpp) for the same reason.
#include "../sycl_runtime/internal.hpp"
#include <algorithm>

using strata::sycl_runtime::current_device;
using strata::sycl_runtime::fail;
using strata::sycl_runtime::from_exception;
using strata::sycl_runtime::registry;
using strata::sycl_runtime::set_last_error;
using strata::sycl_runtime::stream_or_default;

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode) {
    strata_cuda_stream* s = stream_or_default(stream);
    if (s->capturing) return fail(cudaErrorStreamCaptureUnsupported);
    try {
        auto g = std::make_unique<strata_cuda_graph>();
        g->graph = std::make_unique<syclexp::command_graph<syclexp::graph_state::modifiable>>(
            registry().context, registry().devices[s->device]);
        g->device = s->device;
        // Drain the queue first: a recording begun while earlier work (an executable graph and the barrier an
        // event record put after it) is still in flight produced a graph that never started when launched later
        // (measured on the A770, Level Zero v2: the verify window captured after an async commit; the windows
        // captured on an idle queue ran).  Captures are rare (once per window size), so the wait costs nothing.
        s->queue.wait();
        // begin_recording throws when the runtime cannot record this queue (already
        // recording, an unsupported backend) - exactly the fallback trigger.
        g->graph->begin_recording(s->queue);
        g->recording.push_back(s);
        s->capturing = true;
        s->graph = g.release();
        if (strata::sycl_runtime::trace_on())
            strata::sycl_runtime::trace_event("capture begins on queue " + std::to_string((uintptr_t) &s->queue % 100000));
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        from_exception(e);
        return fail(cudaErrorStreamCaptureUnsupported);
    }
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* graph) {
    if (!graph) return fail(cudaErrorInvalidValue);
    *graph = nullptr;
    strata_cuda_stream* s = stream_or_default(stream);
    if (!s->capturing || !s->graph) return fail(cudaErrorStreamCaptureInvalidated);
    try {
        if (strata::sycl_runtime::trace_on()) strata::sycl_runtime::trace_end_marker(s->queue);
        s->graph->graph->end_recording(s->queue);
        s->capturing = false;
        if (strata::sycl_runtime::trace_on())
            strata::sycl_runtime::trace_event("capture ends: graph " + std::to_string((uintptr_t) s->graph % 100000));
        *graph = s->graph;
        s->graph = nullptr;
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        s->capturing = false;
        delete s->graph;
        s->graph = nullptr;
        from_exception(e);
        return fail(cudaErrorStreamCaptureInvalidated);
    }
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream, cudaStreamCaptureStatus* status) {
    if (!status) return fail(cudaErrorInvalidValue);
    *status = stream_or_default(stream)->capturing ? cudaStreamCaptureStatusActive : cudaStreamCaptureStatusNone;
    return set_last_error(cudaSuccess);
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, unsigned long long) {
    if (!exec || !graph || !graph->graph) return fail(cudaErrorInvalidValue);
    *exec = nullptr;
    try {
        auto e = std::make_unique<strata_cuda_graph_exec>();
        e->graph = std::make_unique<syclexp::command_graph<syclexp::graph_state::executable>>(
            graph->graph->finalize());
        e->device = graph->device;
        *exec = e.release();
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e2) {
        return from_exception(e2);
    }
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t graph, cudaGraphNode_t* errorNode,
                                 char* logBuffer, size_t bufferSize) {
    if (errorNode) *errorNode = nullptr;
    if (logBuffer && bufferSize) logBuffer[0] = '\0';
    return cudaGraphInstantiate(exec, graph, 0ull);
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t exec, cudaStream_t stream) {
    strata::sycl_runtime::note_submit(stream_or_default(stream), "graph", __builtin_return_address(0));
    if (!exec || !exec->graph) return fail(cudaErrorInvalidValue);
    try {
        if (strata::sycl_runtime::trace_on())
            strata::sycl_runtime::trace_event("launch exec " + std::to_string((uintptr_t) exec % 100000) + " on queue " +
                                              std::to_string((uintptr_t) &stream_or_default(stream)->queue % 100000));
        stream_or_default(stream)->queue.ext_oneapi_graph(*exec->graph);
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaGraphUpload(cudaGraphExec_t exec, cudaStream_t) {
    // Level Zero builds the executable graph at finalize; there is nothing to upload.
    return exec && exec->graph ? set_last_error(cudaSuccess) : fail(cudaErrorInvalidValue);
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    delete graph;
    return set_last_error(cudaSuccess);
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t exec) {
    delete exec;
    return set_last_error(cudaSuccess);
}

// The installed graph extension exposes recorded nodes and their command types.
// Cache wrappers so CUDA-shaped handles remain stable until graph destruction.
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) {
    if (!graph || !graph->graph || !numNodes) return fail(cudaErrorInvalidValue);
    try {
        const auto recorded = graph->graph->get_nodes();
        if (graph->nodes.empty()) {
            for (const auto& node : recorded)
                graph->nodes.push_back(std::make_unique<strata_cuda_graph_node>(strata_cuda_graph_node{node}));
        }
        if (nodes) {
            const size_t copied = std::min(*numNodes, graph->nodes.size());
            for (size_t i = 0; i < copied; ++i) nodes[i] = graph->nodes[i].get();
        }
        *numNodes = recorded.size();
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* type) {
    if (!node || !type) return fail(cudaErrorInvalidValue);
    try {
        switch (node->node.get_type()) {
            case syclexp::node_type::kernel: *type = cudaGraphNodeTypeKernel; break;
            case syclexp::node_type::memcpy: *type = cudaGraphNodeTypeMemcpy; break;
            case syclexp::node_type::memset: *type = cudaGraphNodeTypeMemset; break;
            case syclexp::node_type::host_task: *type = cudaGraphNodeTypeHost; break;
            case syclexp::node_type::subgraph: *type = cudaGraphNodeTypeGraph; break;
            case syclexp::node_type::empty: *type = cudaGraphNodeTypeEmpty; break;
            default: return fail(cudaErrorUnknown);
        }
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

// SYCL does not expose the original CUDA launch parameters.
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t, cudaKernelNodeParams*) {
    return fail(cudaErrorUnknown);
}

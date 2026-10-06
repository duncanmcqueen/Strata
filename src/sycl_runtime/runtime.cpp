// src/sycl_runtime/runtime.cpp - devices, errors, streams, events of the CUDA shim.
//
// The device list is the Level Zero GPUs of ONEAPI_DEVICE_SELECTOR (honored by the
// SYCL runtime's device filtering) that offer a 32-wide sub-group: Strata's kernels
// assume 32-lane warps, so a GPU without sub-group 32 never enters the list and the
// backend refuses to start on it.  The compute capability reported is a fixed
// 8.0 sentinel: high enough that every `cc >= 75` portable path is taken and the
// pre-sm_75 rejections stay silent, below 9.0 so the sm_90 thread-block-cluster
// paths (sampler.cu, qsa_select.cu) are never selected on a card that has no
// clusters.  The kernels being ported own their launch shape, so the sentinel only
// matters to host-side branches, each re-checked when its file is ported.
#include "../sycl_runtime/internal.hpp"

#include <algorithm>
#include <cstdlib>
#include <cxxabi.h>
#include <chrono>
#include <map>
#include <thread>
#include <tuple>
#include <vector>
#include <dlfcn.h>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>
#include <cstdio>
#include <cstring>

namespace {

thread_local cudaError_t g_last_error = cudaSuccess;
thread_local int g_current_device = 0;

// Async (queue) errors have no thread of their own: a sticky process-wide slot that
// cudaGetLastError drains when the thread-local slot is clean.  The message goes to
// stderr at raise time; cudaGetErrorString can only return the generic text.
std::atomic<int> g_async_error{0};

void async_handler(sycl::exception_list list) {
    for (const std::exception_ptr& p : list) {
        try {
            std::rethrow_exception(p);
        } catch (const sycl::exception& e) {
            std::fprintf(stderr, "strata sycl: async device error: %s\n", e.what());
            g_async_error.store(static_cast<int>(cudaErrorUnknown), std::memory_order_release);
        }
    }
}

sycl::queue make_queue(const sycl::context& context, const sycl::device& device) {
    // Profiling on every queue: cudaEventElapsedTime reads command timestamps, and an
    // event recorded on a non-profiling queue would silently report garbage.
    return sycl::queue(context, device, async_handler,
                       {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
}

bool has_subgroup32(const sycl::device& d) {
    const auto sizes = d.get_info<sycl::info::device::sub_group_sizes>();
    for (size_t s : sizes)
        if (s == 32) return true;
    return false;
}

}  // namespace

namespace strata::sycl_runtime {

std::atomic<int64_t> g_allocated_bytes{0};

Registry& registry() {
    static Registry reg = [] {
        Registry r;
        // Enumerate Level Zero GPUs directly (not the default selector) so the device
        // index Strata sees is stable and matches the tools/sycl/versions.txt table.
        // ONEAPI_DEVICE_SELECTOR is applied by the SYCL runtime at enumeration, so a
        // pinned run sees exactly the pinned card as device 0.
        for (const sycl::platform& p : sycl::platform::get_platforms()) {
            if (p.get_backend() != sycl::backend::ext_oneapi_level_zero) continue;
            for (const sycl::device& d : p.get_devices(sycl::info::device_type::gpu)) {
                if (!has_subgroup32(d)) {
                    std::fprintf(stderr,
                                 "strata sycl: excluding '%s': no 32-wide sub-group (Strata's kernels "
                                 "assume 32-lane warps)\n",
                                 d.get_info<sycl::info::device::name>().c_str());
                    continue;
                }
                r.devices.push_back(d);
            }
        }
        if (r.devices.empty())
            throw sycl::exception(sycl::make_error_code(sycl::errc::runtime),
                                  "strata sycl: no Level Zero GPU with 32-wide sub-group found "
                                  "(check ONEAPI_DEVICE_SELECTOR and the driver)");
        r.context = sycl::context(r.devices, async_handler);
        r.default_streams.reserve(r.devices.size());
        for (size_t i = 0; i < r.devices.size(); ++i) {
            auto s = std::make_unique<strata_cuda_stream>();
            s->queue = make_queue(r.context, r.devices[i]);
            s->device = static_cast<int>(i);
            r.queues.push_back(&s->queue);  // registry() is still private here; no mutex needed yet
            r.default_streams.push_back(std::move(s));
        }
        return r;
    }();
    return reg;
}

std::mutex& queues_mutex() {
    static std::mutex m;
    return m;
}

void register_queue(sycl::queue* q) {
    std::lock_guard<std::mutex> lock(queues_mutex());
    registry().queues.push_back(q);
}

void unregister_queue(sycl::queue* q) {
    std::lock_guard<std::mutex> lock(queues_mutex());
    auto& queues = registry().queues;
    queues.erase(std::remove(queues.begin(), queues.end(), q), queues.end());
}

void register_blocking_stream(strata_cuda_stream* s) {
    std::lock_guard<std::mutex> lock(queues_mutex());
    registry().blocking_streams.push_back(s);
}

void unregister_blocking_stream(strata_cuda_stream* s) {
    std::lock_guard<std::mutex> lock(queues_mutex());
    auto& v = registry().blocking_streams;
    v.erase(std::remove(v.begin(), v.end(), s), v.end());
}

void wait_blocking_streams(int device) {
    std::vector<strata_cuda_stream*> streams;
    {
        std::lock_guard<std::mutex> lock(queues_mutex());
        streams = registry().blocking_streams;
    }
    for (strata_cuda_stream* s : streams)
        if (s->device == device && !s->capturing) s->queue.wait();
}

int current_device() { return g_current_device; }

cudaError_t set_current_device(int device) {
    registry();  // force enumeration (and its failure) at the first device call
    if (device < 0 || device >= static_cast<int>(registry().devices.size()))
        return fail(cudaErrorInvalidValue);
    g_current_device = device;
    return cudaSuccess;
}

strata_cuda_stream* default_stream(int device) { return registry().default_streams[device].get(); }

strata_cuda_stream* stream_or_default(cudaStream_t stream) {
    return stream ? stream : default_stream(current_device());
}

cudaError_t set_last_error(cudaError_t error) {
    g_last_error = error;
    return error;
}

cudaError_t fail(cudaError_t error) { return set_last_error(error); }

cudaError_t from_exception(const sycl::exception& e) {
    std::fprintf(stderr, "strata sycl: %s\n", e.what());
    if (e.code() == sycl::errc::memory_allocation) return fail(cudaErrorMemoryAllocation);
    return fail(cudaErrorUnknown);
}

void note_async_error(const char* what) {
    std::fprintf(stderr, "strata sycl: %s\n", what);
    g_async_error.store(static_cast<int>(cudaErrorUnknown), std::memory_order_release);
}

// The public bridge (include/strata/sycl_runtime/queue_bridge.hpp).
// ---- STRATA_SYCL_TRACE=1: a hang locator.  Every kernel launch asks for its queue here first, so before handing
// it out this enqueues a one-work-item kernel that stores the launch's sequence number into host USM (an uncached
// store: the host sees it while the stream is still running) and remembers who asked (the caller's address).  At exit
// the last number the GPU reached and the launches around it are printed with their symbols: the kernel after the last
// reached number is the one that did not finish.  Off by default; it doubles the launches.
namespace {
struct Trace {
    uint32_t* reached = nullptr;              // host USM, written by the GPU
    std::vector<void*> callers;               // index n-1: the caller of launch n
    std::vector<std::string> events;          // host-side graph events, each with the launch count at the time
    std::vector<uint64_t> samples;            // profile: samples per launch index
    uint64_t total = 0, busy = 0;
    std::mutex mu;
};
Trace* trace_state() {
    static Trace* t = [] () -> Trace* {
        const char* e = std::getenv("STRATA_SYCL_TRACE");
        if (e == nullptr || e[0] == '0') return nullptr;
        auto* tr = new Trace;
        tr->reached = sycl::malloc_host<uint32_t>(16, registry().context);
        *(volatile uint32_t*) tr->reached = 0;
        // STRATA_SYCL_TRACE=profile: a host thread samples the launch the GPU last reached every ~20 us, so each
        // sample is GPU time spent in (or launching) that kernel; at exit the callers are ranked by samples.
        if (std::string(e) == "profile") {
            std::thread([tr] {
                uint32_t last = 0;
                for (;;) {
                    const uint32_t r = *(volatile uint32_t*) tr->reached & 0x7fffffffu;
                    if (r != 0) {
                        std::lock_guard<std::mutex> lock(tr->mu);
                        if (tr->samples.size() <= r) tr->samples.resize(r + 1024, 0);
                        tr->samples[r] += 1;
                        tr->total += 1;
                        tr->busy += (r != last);
                    }
                    last = r;
                    std::this_thread::sleep_for(std::chrono::microseconds(20));
                }
            }).detach();
        }
        std::atexit([] {
            Trace* x = trace_state();
            const uint32_t r_raw = *(volatile uint32_t*) x->reached;
            const uint32_t r = r_raw & 0x7fffffffu;
            std::fprintf(stderr, "strata sycl trace: %zu launches recorded; the GPU reached %u%s\n", x->callers.size(),
                         r, (r_raw & 0x80000000u) ? " (an end-of-graph marker: that graph finished)" : "");
            if (x->total > 0) {   // the profile: samples by caller (launch index -> its caller's symbol)
                std::map<std::string, uint64_t> by;
                for (size_t i = 1; i < x->samples.size() && i <= x->callers.size(); ++i) {
                    if (x->samples[i] == 0) continue;
                    Dl_info info{};
                    const char* name = "?";
                    if (dladdr(x->callers[i - 1], &info) && info.dli_sname) name = info.dli_sname;
                    int st = 0;
                    char* dem = abi::__cxa_demangle(name, nullptr, nullptr, &st);
                    std::string k = dem ? dem : name;
                    std::free(dem);
                    if (!info.dli_sname) {   // not exported: the module and offset, for addr2line
                        char b[64];
                        std::snprintf(b, sizeof b, "+0x%zx", (size_t) ((char*) x->callers[i - 1] - (char*) info.dli_fbase));
                        k = std::string(info.dli_fname ? info.dli_fname : "?") + b;
                    }
                    const size_t paren = k.find('(');
                    if (paren != std::string::npos) k.resize(paren);
                    by[k] += x->samples[i];
                }
                std::vector<std::pair<uint64_t, std::string>> v;
                for (auto& [k, n] : by) v.push_back({n, k});
                std::sort(v.rbegin(), v.rend());
                std::fprintf(stderr, "strata sycl profile: %llu samples (~20 us each)\n", (unsigned long long) x->total);
                for (size_t i = 0; i < v.size() && i < 30; ++i)
                    std::fprintf(stderr, "  %5.1f%%  %8.1f ms  %s\n", 100.0 * v[i].first / x->total, v[i].first * 0.02,
                                 v[i].second.c_str());
            }
            const size_t ne = x->events.size();
            for (size_t i = ne > 12 ? ne - 12 : 0; i < ne; ++i) std::fprintf(stderr, "  event: %s\n", x->events[i].c_str());
            for (uint32_t i = r > 3 ? r - 3 : 1; i <= r + 3 && i <= x->callers.size(); ++i) {
                Dl_info info{};
                const char* name = "?";
                if (dladdr(x->callers[i - 1], &info) && info.dli_sname) name = info.dli_sname;
                int st = 0;
                char* dem = abi::__cxa_demangle(name, nullptr, nullptr, &st);
                std::fprintf(stderr, "  %s launch %u: %s  (%s+0x%zx)\n", i == r + 1 ? "->" : "  ", i, dem ? dem : name,
                             info.dli_fname ? info.dli_fname : "?",
                             (size_t) ((char*) x->callers[i - 1] - (char*) info.dli_fbase));
                std::free(dem);
            }
        });
        return tr;
    }();
    return t;
}
}  // namespace

// ---- STRATA_SYCL_QSTATS=1: which streams submit what.  On the A770 work on two queues does not overlap: a kernel on
// one queue while another queue's kernel runs costs ~300-380 us of switching (measured), so the decode should keep
// its GPU work on one queue.  The table (stream, kind, caller, count) is printed at exit.
namespace {
struct QStats {
    std::mutex mu;
    std::map<std::tuple<const void*, std::string, void*>, long> m;
};
QStats* qstats() {
    static QStats* q = [] () -> QStats* {
        const char* e = std::getenv("STRATA_SYCL_QSTATS");
        if (e == nullptr || e[0] == '0' || e[0] == '\0') return nullptr;
        auto* x = new QStats;
        std::atexit([] {
            QStats* x = qstats();
            std::lock_guard<std::mutex> lock(x->mu);
            std::fprintf(stderr, "strata sycl qstats: %zu (stream, kind) pairs\n", x->m.size());
            for (const auto& [k, n] : x->m) {
                void* const caller = std::get<2>(k);
                Dl_info info{};
                const char* name = "?";
                if (dladdr(caller, &info) && info.dli_sname) name = info.dli_sname;
                int st = 0;
                char* dem = abi::__cxa_demangle(name, nullptr, nullptr, &st);
                std::fprintf(stderr, "  stream %p %-8s %8ld  from %.110s (%s+0x%zx)\n", std::get<0>(k),
                             std::get<1>(k).c_str(), n, dem ? dem : name, info.dli_fname ? info.dli_fname : "?",
                             (size_t) ((char*) caller - (char*) info.dli_fbase));
                std::free(dem);
            }
        });
        return x;
    }();
    return q;
}
}  // namespace

void note_submit(const void* stream, const char* kind, void* caller) {
    QStats* q = qstats();
    if (!q) return;
    std::lock_guard<std::mutex> lock(q->mu);
    ++q->m[{stream, kind, caller}];
}

void trace_event(const std::string& what) {
    if (Trace* tr = trace_state()) {
        std::lock_guard<std::mutex> lock(tr->mu);
        tr->events.push_back(what + " (launches so far " + std::to_string(tr->callers.size()) + ")");
    }
}
bool trace_on() { return trace_state() != nullptr; }
// a marker after the last kernel recorded so far on q: 0x80000000 | launches so far ("everything before ran")
void trace_end_marker(sycl::queue& q) {
    Trace* tr = trace_state();
    if (!tr) return;
    uint32_t n;
    {
        std::lock_guard<std::mutex> lock(tr->mu);
        n = 0x80000000u | (uint32_t) tr->callers.size();
    }
    uint32_t* reached = tr->reached;
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;
    using wr = intelex::write_hint_key::value_t<
        intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L3>>;
    q.single_task([=]() {
        syclex::annotated_ptr<uint32_t, decltype(syclex::properties(wr{}))> a(reached);
        a[0] = n;
    });
}

// STRATA_SYCL_DEBUG_SKIP=name1,name2 (debugging only: the results are wrong): while a stream is being captured, a
// caller whose symbol contains one of the names gets a scratch queue instead, so its kernels stay out of the graph.
// The change in a window's time is then that caller's cost inside the real graph.
namespace {
sycl::queue* debug_skip_queue(void* caller) {
    static const std::vector<std::string> names = [] {
        std::vector<std::string> v;
        const char* e = std::getenv("STRATA_SYCL_DEBUG_SKIP");
        for (std::string rest = e ? e : ""; !rest.empty();) {
            const size_t c = rest.find(',');
            v.push_back(rest.substr(0, c));
            rest = c == std::string::npos ? "" : rest.substr(c + 1);
        }
        return v;
    }();
    if (names.empty()) return nullptr;
    Dl_info info{};
    if (!dladdr(caller, &info) || !info.dli_sname) return nullptr;
    int st = 0;
    char* dem = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &st);
    const std::string sym = dem ? dem : info.dli_sname;
    std::free(dem);
    for (const auto& n : names)
        if (!n.empty() && sym.find(n) != std::string::npos) {
            static sycl::queue* scratch = new sycl::queue(registry().context, registry().devices[current_device()],
                                                          sycl::property::queue::in_order{});
            return scratch;
        }
    return nullptr;
}
}  // namespace

__attribute__((noinline)) sycl::queue& queue_from_stream(void* stream) {
    strata_cuda_stream* const sp = stream_or_default(static_cast<cudaStream_t>(stream));
    if (sp->capturing)
        if (sycl::queue* skip = debug_skip_queue(__builtin_return_address(0))) return *skip;
    sycl::queue& q = sp->queue;
    note_submit(sp, sp->capturing ? "kernel@c" : "kernel", __builtin_return_address(0));
    if (Trace* tr = trace_state()) {
        uint32_t n;
        {
            std::lock_guard<std::mutex> lock(tr->mu);
            tr->callers.push_back(__builtin_return_address(0));
            n = (uint32_t) tr->callers.size();
        }
        uint32_t* reached = tr->reached;
        namespace syclex = sycl::ext::oneapi::experimental;
        namespace intelex = sycl::ext::intel::experimental;
        using wr = intelex::write_hint_key::value_t<
            intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L3>>;
        q.single_task([=]() {
            syclex::annotated_ptr<uint32_t, decltype(syclex::properties(wr{}))> a(reached);
            a[0] = n;
        });
    }
    return q;
}
sycl::device device_current() { return registry().devices[current_device()]; }
sycl::context context_shared() { return registry().context; }

}  // namespace strata::sycl_runtime

using strata::sycl_runtime::current_device;
using strata::sycl_runtime::fail;
using strata::sycl_runtime::from_exception;
using strata::sycl_runtime::registry;
using strata::sycl_runtime::set_last_error;
using strata::sycl_runtime::stream_or_default;

// ---- errors ---------------------------------------------------------------------

cudaError_t cudaGetLastError() {
    const cudaError_t thread = g_last_error;
    g_last_error = cudaSuccess;
    if (thread != cudaSuccess) return thread;
    return static_cast<cudaError_t>(g_async_error.exchange(0, std::memory_order_acq_rel));
}

cudaError_t cudaPeekAtLastError() {
    if (g_last_error != cudaSuccess) return g_last_error;
    return static_cast<cudaError_t>(g_async_error.load(std::memory_order_acquire));
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess: return "no error";
        case cudaErrorInvalidValue: return "invalid argument";
        case cudaErrorMemoryAllocation: return "out of memory";
        case cudaErrorNoDevice: return "no Level Zero GPU with 32-wide sub-group";
        case cudaErrorNotReady: return "device not ready";
        case cudaErrorPeerAccessAlreadyEnabled: return "peer access already enabled";
        case cudaErrorStreamCaptureUnsupported: return "operation not supported during stream capture";
        case cudaErrorStreamCaptureInvalidated: return "stream capture invalidated";
        case cudaErrorUnknown:
        default: return "SYCL error (see stderr for the device message)";
    }
}

// ---- devices ----------------------------------------------------------------------

cudaError_t cudaGetDeviceCount(int* count) {
    if (!count) return fail(cudaErrorInvalidValue);
    try {
        *count = static_cast<int>(registry().devices.size());
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        *count = 0;
        return from_exception(e);
    }
}

cudaError_t cudaSetDevice(int device) { return strata::sycl_runtime::set_current_device(device); }

cudaError_t cudaGetDevice(int* device) {
    if (!device) return fail(cudaErrorInvalidValue);
    *device = current_device();
    return set_last_error(cudaSuccess);
}

cudaError_t cudaInitDevice(int device, int, int) { return strata::sycl_runtime::set_current_device(device); }

namespace {

const sycl::device* device_checked(int device, cudaError_t* error) {
    registry();
    if (device < 0 || device >= static_cast<int>(registry().devices.size())) {
        *error = fail(cudaErrorInvalidValue);
        return nullptr;
    }
    *error = cudaSuccess;
    return &registry().devices[device];
}

// Xe-core count: slices x sub-slices per slice (a sub-slice IS an Xe core on Arc).
// Falls back to the EU count if the platform predates the extension queries.
int xe_core_count(const sycl::device& d) {
    using namespace sycl::ext::intel::info::device;
    try {
        const int slices = d.get_info<gpu_slices>();
        const int subslices = d.get_info<gpu_subslices_per_slice>();
        if (slices > 0 && subslices > 0) return slices * subslices;
    } catch (const sycl::exception&) {
    }
    return static_cast<int>(d.get_info<sycl::info::device::max_compute_units>());
}

cudaError_t fill_properties(cudaDeviceProp* prop, const sycl::device& d) {
    std::memset(prop, 0, sizeof(*prop));
    const std::string name = d.get_info<sycl::info::device::name>();
    std::strncpy(prop->name, name.c_str(), sizeof(prop->name) - 1);
    prop->totalGlobalMem = d.get_info<sycl::info::device::global_mem_size>();
    prop->l2CacheSize = static_cast<int>(d.get_info<sycl::info::device::global_mem_cache_size>());
    const auto local_mem = d.get_info<sycl::info::device::local_mem_size>();
    prop->sharedMemPerBlock = local_mem;
    prop->sharedMemPerMultiprocessor = local_mem;
    prop->warpSize = 32;  // guaranteed by the enumeration's sub-group filter
    prop->maxThreadsPerBlock = static_cast<int>(d.get_info<sycl::info::device::max_work_group_size>());
    const auto item_sizes = d.get_info<sycl::info::device::max_work_item_sizes<3>>();
    prop->maxThreadsDim[0] = static_cast<int>(item_sizes[0]);
    prop->maxThreadsDim[1] = static_cast<int>(item_sizes[1]);
    prop->maxThreadsDim[2] = static_cast<int>(item_sizes[2]);
    prop->maxGridSize[0] = 0x7fffffff;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->clockRate = static_cast<int>(d.get_info<sycl::info::device::max_clock_frequency>()) * 1000;  // MHz -> KHz
    prop->major = 8;  // the sentinel: >= 7.5 portable paths on, sm_90 cluster paths off
    prop->minor = 0;
    prop->multiProcessorCount = xe_core_count(d);
    prop->maxThreadsPerMultiProcessor =
        static_cast<int>(d.get_info<sycl::info::device::max_work_group_size>());
    prop->integrated = d.get_info<sycl::info::device::host_unified_memory>() ? 1 : 0;
    prop->cooperativeLaunch = 0;
    return cudaSuccess;
}

}  // namespace

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    if (!prop) return fail(cudaErrorInvalidValue);
    cudaError_t error = cudaSuccess;
    const sycl::device* d = device_checked(device, &error);
    if (error != cudaSuccess) return error;
    try {
        return set_last_error(fill_properties(prop, *d));
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device) {
    if (!value) return fail(cudaErrorInvalidValue);
    cudaError_t error = cudaSuccess;
    const sycl::device* d = device_checked(device, &error);
    if (error != cudaSuccess) return error;
    try {
        switch (attr) {
            case cudaDevAttrComputeCapabilityMajor: *value = 8; break;  // sentinel, see fill_properties
            case cudaDevAttrComputeCapabilityMinor: *value = 0; break;
            case cudaDevAttrMultiProcessorCount: *value = xe_core_count(*d); break;
            case cudaDevAttrClockRate:
                *value = static_cast<int>(d->get_info<sycl::info::device::max_clock_frequency>()) * 1000;
                break;
            case cudaDevAttrWarpSize: *value = 32; break;
            case cudaDevAttrMaxSharedMemoryPerBlock:
            case cudaDevAttrMaxSharedMemoryPerBlockOptin:
            case cudaDevAttrMaxSharedMemoryPerMultiprocessor:
                *value = static_cast<int>(d->get_info<sycl::info::device::local_mem_size>());
                break;
            case cudaDevAttrReservedSharedMemoryPerBlock: *value = 0; break;
            case cudaDevAttrMaxThreadsPerBlock:
                *value = static_cast<int>(d->get_info<sycl::info::device::max_work_group_size>());
                break;
            case cudaDevAttrIntegrated: *value = d->get_info<sycl::info::device::host_unified_memory>() ? 1 : 0; break;
            case cudaDevAttrCooperativeLaunch: *value = 0; break;
            default: return fail(cudaErrorInvalidValue);
        }
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaDeviceSynchronize() {
    try {
        // The whole device, not one stream: every queue this process made for it.
        std::vector<sycl::queue*> queues;
        {
            std::lock_guard<std::mutex> lock(strata::sycl_runtime::queues_mutex());
            queues = registry().queues;
        }
        for (sycl::queue* q : queues) q->wait();
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaDeviceCanAccessPeer(int* canAccess, int device, int peerDevice) {
    if (!canAccess) return fail(cudaErrorInvalidValue);
    *canAccess = 0;  // peer copies are out of scope for v1 (cudaMemcpyPeerAsync refuses)
    (void) device;
    (void) peerDevice;
    return set_last_error(cudaSuccess);
}

cudaError_t cudaDeviceEnablePeerAccess(int, unsigned int) {
    // Reporting "already enabled" would be a lie Strata acts on; refuse plainly.
    return fail(cudaErrorInvalidValue);
}

cudaError_t cudaRuntimeGetVersion(int* version) {
    if (!version) return fail(cudaErrorInvalidValue);
    *version = 12080;  // the CUDA shape the sources were written against; not a SYCL version
    return set_last_error(cudaSuccess);
}

cudaError_t cudaDriverGetVersion(int* version) {
    if (!version) return fail(cudaErrorInvalidValue);
    *version = 12080;
    return set_last_error(cudaSuccess);
}

// ---- streams ----------------------------------------------------------------------

cudaError_t cudaStreamCreate(cudaStream_t* stream) { return cudaStreamCreateWithFlags(stream, cudaStreamDefault); }

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    if (!stream) return fail(cudaErrorInvalidValue);
    try {
        auto s = std::make_unique<strata_cuda_stream>();
        const int device = current_device();
        s->queue = make_queue(registry().context, registry().devices[device]);
        s->device = device;
        s->blocking = (flags & cudaStreamNonBlocking) == 0;
        strata_cuda_stream* raw = s.release();
        strata::sycl_runtime::register_queue(&raw->queue);
        if (raw->blocking) strata::sycl_runtime::register_blocking_stream(raw);
        *stream = raw;
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (!stream) return set_last_error(cudaSuccess);  // destroying the default stream is a no-op
    if (stream->capturing) return fail(cudaErrorInvalidValue);
    strata::sycl_runtime::unregister_queue(&stream->queue);
    if (stream->blocking) strata::sycl_runtime::unregister_blocking_stream(stream);
    try {
        stream->queue.wait();
    } catch (const sycl::exception& e) {
        delete stream;
        return from_exception(e);
    }
    delete stream;
    return set_last_error(cudaSuccess);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    try {
        stream_or_default(stream)->queue.wait();
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    try {
        return set_last_error(stream_or_default(stream)->queue.ext_oneapi_empty() ? cudaSuccess : cudaErrorNotReady);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int) {
    if (!event || !event->recorded) return set_last_error(cudaSuccess);  // an unrecorded event is complete
    try {
        strata_cuda_stream* const s = stream_or_default(stream);
        // Outside a capture, the host waits instead of the GPU.  A device-side barrier on an event of another queue
        // is a semaphore wait on the engine: while the prompt path streams experts over a slow link it held the
        // compute engine for longer than the xe driver's 5 s job timeout (the engine was reset mid-prompt and the
        // reply came out as garbage), and two queues waiting on each other this way deadlocked.  The event's work is
        // already submitted, so a host wait cannot deadlock; the queue's own order covers the rest.
        static const bool device_waits = std::getenv("STRATA_SYCL_DEVICE_EVENT_WAITS") != nullptr;
        if (!s->capturing && !device_waits) {
            if (event->event.get_info<sycl::info::event::command_execution_status>() !=
                sycl::info::event_command_status::complete)
                event->event.wait();
            return set_last_error(cudaSuccess);
        }
        s->queue.ext_oneapi_submit_barrier({event->event});
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    if (!fn) return fail(cudaErrorInvalidValue);
    strata_cuda_stream* s = stream_or_default(stream);
    strata::sycl_runtime::note_submit(s, "hostfn", __builtin_return_address(0));
    if (s->capturing) {
        // DPC++ does not record host tasks into command graphs.  Refuse so Strata's
        // non-captured fallback runs (the guide's graph rule for exactly this case).
        return fail(cudaErrorStreamCaptureUnsupported);
    }
    try {
        s->queue.submit([fn, userData](sycl::handler& h) { h.host_task([fn, userData] { fn(userData); }); });
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

// ---- events ------------------------------------------------------------------------

cudaError_t cudaEventCreate(cudaEvent_t* event) { return cudaEventCreateWithFlags(event, cudaEventDefault); }

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    if (!event) return fail(cudaErrorInvalidValue);
    auto e = std::make_unique<strata_cuda_event>();
    e->timing = (flags & cudaEventDisableTiming) == 0;
    e->device = current_device();
    *event = e.release();
    return set_last_error(cudaSuccess);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event;
    return set_last_error(cudaSuccess);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (!event) return fail(cudaErrorInvalidValue);
    try {
        // A barrier after everything submitted so far; recorded into the graph when
        // the queue is capturing, which is exactly the dependency node CUDA makes.
        event->event = stream_or_default(stream)->queue.ext_oneapi_submit_barrier();
        event->recorded = true;
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    if (!event || !event->recorded) return set_last_error(cudaSuccess);
    try {
        event->event.wait();
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    // The graph rules require waits to poll HERE, never memory: on Level Zero a host
    // loop that only reads can starve submission.  An unrecorded event is complete.
    if (!event || !event->recorded) return set_last_error(cudaSuccess);
    try {
        const auto status = event->event.get_info<sycl::info::event::command_execution_status>();
        return set_last_error(status == sycl::info::event_command_status::complete ? cudaSuccess
                                                                                   : cudaErrorNotReady);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (!ms || !start || !end) return fail(cudaErrorInvalidValue);
    if (!start->recorded || !end->recorded) return fail(cudaErrorInvalidValue);
    if (!start->timing || !end->timing) {
        *ms = 0.0f;
        return set_last_error(cudaSuccess);
    }
    try {
        const uint64_t t0 = start->event.get_profiling_info<sycl::info::event_profiling::command_start>();
        const uint64_t t1 = end->event.get_profiling_info<sycl::info::event_profiling::command_end>();
        *ms = static_cast<float>(t1 - t0) * 1e-6f;  // ns -> ms
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

// ---- function attributes / occupancy -------------------------------------------------

cudaError_t cudaFuncSetAttribute(const void*, cudaFuncAttribute attribute, int value) {
    if (attribute != cudaFuncAttributeMaxDynamicSharedMemorySize &&
        attribute != cudaFuncAttributePreferredSharedMemoryCarveout) {
        return fail(cudaErrorInvalidValue);
    }
    try {
        const auto local_mem = static_cast<int>(
            registry().devices[current_device()].get_info<sycl::info::device::local_mem_size>());
        // There is no opt-in to set on a SYCL launch; validate the request the way the
        // CUDA call would and refuse loudly when it cannot hold.
        if (value < 0 || value > local_mem) return fail(cudaErrorInvalidValue);
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaFuncGetAttributes(cudaFuncAttributes*, const void*) {
    return fail(cudaErrorUnknown);  // see the header: no function-pointer mapping in SYCL
}

cudaError_t cudaFuncGetName(const char** name, const void*) {
    if (name) *name = "sycl-kernel";
    return set_last_error(cudaSuccess);
}

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int*, const void*, int, size_t) {
    return fail(cudaErrorUnknown);  // revisited with the prefill kernels that call it
}

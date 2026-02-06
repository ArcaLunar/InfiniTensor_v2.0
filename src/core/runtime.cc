#include "core/runtime.h"

namespace infini {
thread_local Context RuntimeObj::tls_context_cache = nullptr;
thread_local std::thread::id RuntimeObj::tls_thread_id;

Runtime &RuntimeObj::getInstance() {
    static Runtime instance = make_ref<RuntimeObj>();
    return instance;
}

RuntimeObj::~RuntimeObj() {
    // 清理所有线程的 Context
    std::unique_lock<std::shared_mutex> lock(ctx_mutex);
    // 只清空map，不手动释放CUDA资源
    // CUDA运行时会在程序退出时自动清理所有资源
    threadContexts.clear();
}

void RuntimeObj::initThreadContext(infiniDevice_t device, int deviceId) {
    auto current_tid = std::this_thread::get_id();
    // 检测线程复用
    if (tls_context_cache && tls_thread_id == current_tid &&
        tls_context_cache->device == device &&
        tls_context_cache->deviceId == deviceId) {
        return; // 已初始化且设备相同，无需重新初始化
    }

    CHECK_INFINI_ERROR(infinirtSetDevice(device, deviceId));

    // 创建新的 stream
    infinirtStream_t stream = nullptr;
    CHECK_INFINI_ERROR(infinirtStreamCreate(&stream));

    // 创建新的 Context
    Context ctx = std::make_shared<ContextObj>();
    ctx->device = device;
    ctx->deviceId = deviceId;
    ctx->stream = stream;
    ctx->workspaceSize = 7ll << 30; // 7GB
    ctx->workspace = nullptr;

    // 更新缓存和全局 map
    tls_context_cache = ctx;
    tls_thread_id = current_tid;

    {
        std::unique_lock<std::shared_mutex> lock(ctx_mutex);
        threadContexts[current_tid] = ctx;
    }
    CHECK_INFINI_ERROR(infinirtMalloc(&ctx->workspace, ctx->workspaceSize));
}

Context RuntimeObj::getCurrentThreadContext() const {
    auto current_tid = std::this_thread::get_id();

    // 检查缓存有效性
    if (tls_context_cache && tls_thread_id == current_tid) {
        return tls_context_cache;
    }

    // 从全局 map 查找
    {
        std::shared_lock<std::shared_mutex> lock(ctx_mutex);
        auto it = threadContexts.find(current_tid);
        if (it != threadContexts.end()) {
            tls_context_cache = it->second;
            tls_thread_id = current_tid;
            return it->second;
        }
    }

    throw std::runtime_error(
        "Thread context not initialized! Call initThreadContext() first.");
}

void RuntimeObj::setCurrentDevice(infiniDevice_t device, int deviceId) {
    auto ctx = getCurrentThreadContext();

    // 如果设备相同，直接返回
    if (ctx->device == device && ctx->deviceId == deviceId) {
        return;
    }

    // 重新初始化 Context（force=true）
    initThreadContext(device, deviceId);
}

void RuntimeObj::init() { CHECK_INFINI_ERROR(infinirtInit()); }

void RuntimeObj::getAllDeviceCount(int *count_array) {
    CHECK_INFINI_ERROR(infinirtGetAllDeviceCount(count_array));
}

void RuntimeObj::run(const Graph &graph) const {
    auto ctx = getCurrentThreadContext();

    IT_ASSERT(graph->checkBeforRun());
    // TODO: 目前仅支持单卡，后续支持多卡
    const auto &kernelRegistry = KernelRegistry::getInstance();
    for (auto &op : graph->getOperators()) {
        auto kernelAttrs =
            KernelAttrs{ctx->device, op->getOpType().underlying()};
        Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
        kernel->compute(op, this);
    }
}

void RuntimeObj::dataMalloc(const Graph &graph) {
    IT_ASSERT(graph->checkBeforRun());
    for (auto &tensor : graph->getTensors()) {
        tensor->dataMalloc(shared_from_this());
    }
}

void *RuntimeObj::allocHost(size_t size) {
    void *ptr;
    CHECK_INFINI_ERROR(infinirtMallocHost(&ptr, size));
    return ptr;
}

void *RuntimeObj::allocDevice(size_t size) {
    void *ptr = nullptr;
    CHECK_INFINI_ERROR(infinirtMalloc(&ptr, size));
    return ptr;
}

void RuntimeObj::deallocHost(void *ptr) {
    CHECK_INFINI_ERROR(infinirtFreeHost(ptr));
}

void RuntimeObj::deallocDevice(void *ptr) {
    CHECK_INFINI_ERROR(infinirtFree(ptr));
}

void RuntimeObj::memcpy(void *dst, const void *src, size_t size,
                        infinirtMemcpyKind_t kind) {
    // 基本指针有效性检查
    if (dst == nullptr || src == nullptr) {
        std::cerr << "[ERROR] memcpy called with null pointer!" << std::endl;
        // 这里应该抛出异常或返回错误，而不是继续
        throw std::runtime_error("Null pointer in memcpy");
    }

    CHECK_INFINI_ERROR(infinirtMemcpy(dst, src, size, kind));
}

void RuntimeObj::memcpyAsync(void *dst, const void *src, size_t size,
                             infinirtMemcpyKind_t kind,
                             infinirtStream_t stream) {
    CHECK_INFINI_ERROR(infinirtMemcpyAsync(dst, src, size, kind, stream));
}

void *RuntimeObj::mallocAsync(size_t size, infinirtStream_t stream) {
    void *ptr = nullptr;
    CHECK_INFINI_ERROR(infinirtMallocAsync(&ptr, size, stream));
    return ptr;
}

void RuntimeObj::freeAsync(void *ptr, infinirtStream_t stream) {
    CHECK_INFINI_ERROR(infinirtFreeAsync(ptr, stream));
}

void RuntimeObj::synchronize() const {
    CHECK_INFINI_ERROR(infinirtDeviceSynchronize());
}

void *RuntimeObj::getWorkspace(size_t size) const {
    auto ctx = getCurrentThreadContext();
    if (!ctx->workspace) {
        throw std::runtime_error(
            "Workspace not initialized! Call initWorkspace() first.");
    }
    return ctx->workspace;
}

size_t RuntimeObj::getWorkspaceSize() const {
    auto ctx = getCurrentThreadContext();
    return ctx->workspaceSize;
}

bool RuntimeObj::isCpu() const {
    auto context = getCurrentThreadContext();
    return context->device == INFINI_DEVICE_CPU;
}

// void RuntimeObj::initWorkspace(size_t size) {
//     auto ctx = getCurrentThreadContext();

//     // 如果已分配且大小足够，直接返回
//     if (ctx->workspace && ctx->workspaceSize >= size) {
//         return;
//     }

//     // 释放旧的 workspace
//     if (ctx->workspace) {
//         infinirtFree(ctx->workspace);
//     }

//     // CPU设备不需要调用setDevice，避免与GPU线程冲突
//     if (ctx->device != INFINI_DEVICE_CPU) {
//         CHECK_INFINI_ERROR(infinirtSetDevice(ctx->device, ctx->deviceId));
//     }

//     // 分配新的 workspace
//     ctx->workspaceSize = size;
//     ctx->workspace = nullptr;
//     CHECK_INFINI_ERROR(infinirtMalloc(&ctx->workspace, size));
// }
} // namespace infini

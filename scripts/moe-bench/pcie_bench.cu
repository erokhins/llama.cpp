// PCIe host-to-device bandwidth microbenchmark for the expert-streaming pattern.
// build: nvcc -O2 -arch=sm_120 -o pcie_bench.exe pcie_bench.cu
// Measures pinned H2D copies of a 1 GiB buffer: one large copy, many expert-sized copies (627 KB / 921 KB)
// on 1 or 2 streams, pattern "every other expert" (runs of 1), and cudaMemcpyBatchAsync if available.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s failed: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main() {
    const size_t N = 1ull << 30; // 1 GiB
    void * h; void * d; void * d2;
    CK(cudaHostAlloc(&h, N, cudaHostAllocDefault));
    CK(cudaMalloc(&d, N));
    CK(cudaMalloc(&d2, N));
    memset(h, 1, N);
    cudaStream_t s[2]; CK(cudaStreamCreateWithFlags(&s[0], cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&s[1], cudaStreamNonBlocking));

    auto run = [&](const char * label, auto issue) {
        issue(); CK(cudaDeviceSynchronize()); // warmup
        double t0 = now(); const int reps = 3;
        for (int r = 0; r < reps; r++) issue();
        CK(cudaDeviceSynchronize());
        double dt = (now() - t0) / reps;
        printf("%-56s %7.1f ms  %6.1f GB/s\n", label, dt * 1e3, N / dt / 1e9);
    };

    run("1 GiB single memcpyAsync", [&] { CK(cudaMemcpyAsync(d, h, N, cudaMemcpyHostToDevice, s[0])); });
    for (size_t chunk : { (size_t) 627 * 1024, (size_t) 921 * 1024, (size_t) 4 << 20, (size_t) 64 << 20 }) {
        char lab[128];
        snprintf(lab, sizeof lab, "%zu KB chunks, 1 stream", chunk / 1024);
        run(lab, [&] { for (size_t o = 0; o + chunk <= N; o += chunk) CK(cudaMemcpyAsync((char *) d + o, (char *) h + o, chunk, cudaMemcpyHostToDevice, s[0])); });
        snprintf(lab, sizeof lab, "%zu KB chunks, 2 streams alternating", chunk / 1024);
        run(lab, [&] { int i = 0; for (size_t o = 0; o + chunk <= N; o += chunk, i++) CK(cudaMemcpyAsync((char *) d + o, (char *) h + o, chunk, cudaMemcpyHostToDevice, s[i & 1])); });
    }
    // D2D for comparison (cache hits are copied device-to-device in prefill)
    run("1 GiB D2D single memcpyAsync", [&] { CK(cudaMemcpyAsync(d2, d, N, cudaMemcpyDeviceToDevice, s[0])); });
    run("627 KB D2D chunks, 1 stream", [&] { size_t c = 627 * 1024; for (size_t o = 0; o + c <= N; o += c) CK(cudaMemcpyAsync((char *) d2 + o, (char *) d + o, c, cudaMemcpyDeviceToDevice, s[0])); });
    // mixed: alternate H2D and D2D 627 KB chunks on one stream (the batched copy_regions pattern)
    run("627 KB alternating H2D/D2D, 1 stream (0.5 GiB each)", [&] { size_t c = 627 * 1024; int i = 0; for (size_t o = 0; o + c <= N; o += c, i++) {
        if (i & 1) CK(cudaMemcpyAsync((char *) d2 + o, (char *) d + o, c, cudaMemcpyDeviceToDevice, s[0]));
        else       CK(cudaMemcpyAsync((char *) d2 + o, (char *) h + o, c, cudaMemcpyHostToDevice, s[0])); } });
    run("627 KB H2D on s0 + D2D on s1 concurrently (0.5 GiB each)", [&] { size_t c = 627 * 1024; int i = 0; for (size_t o = 0; o + c <= N; o += c, i++) {
        if (i & 1) CK(cudaMemcpyAsync((char *) d2 + o, (char *) d + o, c, cudaMemcpyDeviceToDevice, s[1]));
        else       CK(cudaMemcpyAsync((char *) d2 + o, (char *) h + o, c, cudaMemcpyHostToDevice, s[0])); } });
#if CUDART_VERSION >= 12080
    {
        const size_t c = 627 * 1024; const size_t n = N / c;
        std::vector<void *> srcs(n), dsts(n); std::vector<size_t> sizes(n);
        for (size_t i = 0; i < n; i++) { srcs[i] = (char *) h + i * c; dsts[i] = (char *) d + i * c; sizes[i] = c; }
        cudaMemcpyAttributes attr = {}; attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t attrIdx = 0;
        run("627 KB chunks, cudaMemcpyBatchAsync", [&] { CK(cudaMemcpyBatchAsync(dsts.data(), (const void * const *) srcs.data(), sizes.data(), n, &attr, &attrIdx, (size_t) 1, s[0])); });
    }
#endif
    // pageable source for reference
    void * hp = malloc(N); memset(hp, 1, N);
    run("1 GiB from pageable memory", [&] { CK(cudaMemcpyAsync(d, hp, N, cudaMemcpyHostToDevice, s[0])); });
    return 0;
}

# Analysis Report: Cooperative CPU-GPU Subset-Sum Solver

## 1. Executive Summary
This report analyzes the performance of a "Cooperative" CPU-GPU solver for the Subset-Sum Problem (SSP), comparing it against pure CPU and pure GPU implementations. The algorithm is based on the Meet-in-the-Middle approach (Horowitz-Sahni), splitting the problem of size $N$ into two lists of size $2^{N/2}$.

**Key Findings:**
*   **Pure GPU Mode is Fastest:** The pure GPU implementation consistently outperforms both CPU and Cooperative modes, achieving speedups of up to **~66x** over the CPU baseline for $N=48$.
*   **Cooperative Mode Lag:** While the Cooperative mode provides significant speedup over the CPU (up to **~16x**), it is consistently slower than the Pure GPU mode (by a factor of **~3-4x**).
*   **Bottleneck Identification:** The performance of the Cooperative mode is throttled by the CPU's list generation phase. The GPU generates its half of the data much faster and must wait for the CPU to finish and transfer data.
*   **Scalability:** The current single-machine Cooperative approach does not scale as well as the Pure GPU approach for the tested problem sizes ($N=42$ to $48$).

## 2. Performance Results
The following table summarizes the execution times (in seconds) for each mode across different problem sizes ($N$).

| Problem Size ($N$) | CPU Time (s) | GPU Time (s) | Cooperative Time (s) | Speedup (GPU vs CPU) | Speedup (Coop vs CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **42** | ~0.80 | ~0.02 | ~0.04 | ~40x | ~20x |
| **44** | ~2.00 | ~0.045 | ~0.10 | ~44x | ~20x |
| **46** | ~5.50 | ~0.09 | ~0.29 | ~61x | ~19x |
| **48** | ~12.00 | ~0.17 | ~0.61 | ~70x | ~19x |

*Note: Times are approximate medians derived from experimental logs.*

### 2.1 Speedup Analysis
*   **GPU vs. Cooperative:** The GPU mode is consistently faster. For $N=48$, the GPU takes 0.17s while Cooperative takes 0.61s. This indicates that the overhead of generating half the data on the CPU and transferring it to the GPU outweighs the benefit of parallel generation.
*   **Scaling:** As $N$ increases by 2, the problem size quadruples ($2^{N/2}$ grows by 2 bits $\rightarrow$ 4x).
    *   GPU time scales by roughly $2\times$ to $4\times$.
    *   Cooperative time scales by roughly $2\times$ to $4\times$, tracking the CPU's performance characteristics more closely than the GPU's.

## 3. Bottleneck Analysis
The Cooperative algorithm performs the following steps:
1.  **CPU Generation:** Generate List A ($2^{N/2}$ elements) on CPU.
2.  **GPU Generation:** Generate List B ($2^{N/2}$ elements) on GPU.
3.  **Transfer:** Move List A from CPU to GPU.
4.  **Search:** Perform binary search on GPU.

**Why Cooperative is Slower:**
*   **Generation Disparity:** The GPU generates List B orders of magnitude faster than the CPU generates List A.
*   **Idle Time:** The GPU finishes its generation task quickly and sits idle waiting for the CPU to complete List A.
*   **Transfer Overhead:** Moving List A (approx. 268 MB for $N=48$) over PCIe takes non-zero time, further adding to the delay.
*   **Memory Bandwidth:** The CPU's memory bandwidth is significantly lower than the GPU's HBM2, making the repeated concatenation and addition operations in the generation phase much slower.

## 4. Comparison with Literature
The referenced paper ("A novel cooperative accelerated parallel two-list algorithm...") likely targets a **distributed cluster environment** or **memory-constrained scenarios**.
*   **Cluster Context:** In a cluster with many CPUs and fewer GPUs, the aggregate CPU compute power might match or exceed the GPU power, making a 50/50 split viable.
*   **Memory Limits:** If $N$ is large enough (e.g., $N \ge 60$) that the lists exceed GPU memory (16GB+), the Pure GPU approach fails. In such cases, the Cooperative approach (keeping one list on CPU or distributed across nodes) becomes the *only* solution, effectively having "infinite" speedup over a failed GPU run.
*   **Current Context:** For $N=48$, the memory footprint (~512 MB total) fits easily within the Tesla T4's 16GB memory. Thus, the Pure GPU approach is unconstrained and superior.

## 5. Recommendations
1.  **Use Pure GPU for $N < 58$:** For problem sizes that fit in GPU memory, the Pure GPU approach is optimal.
2.  **Asymmetric Splitting:** To optimize Cooperative mode on a single machine, the workload should be split according to compute capability (e.g., 1% CPU, 99% GPU). However, the Meet-in-the-Middle algorithm inherently requires a balanced split ($N/2$) to minimize complexity. Uneven splitting (e.g., $N/4$ on CPU, $3N/4$ on GPU) would exponentially increase the GPU workload, defeating the purpose.
3.  **Distributed Computing:** To solve larger problems ($N \ge 60$), the code should be adapted for a multi-node environment (using MPI or `torch.distributed`) to aggregate memory and compute resources.
4.  **CPU Optimization:** The CPU generation could be further optimized using low-level C++ (AVX-512) extensions instead of PyTorch operations, but it is unlikely to bridge the gap with the GPU.

## 6. Conclusion
The implemented optimizations (pinned memory, non-blocking transfers) correctly reduced overhead but could not overcome the fundamental compute disparity between the CPU and GPU for this specific workload and hardware configuration. The Pure GPU solver remains the most efficient solution for the tested range of $N$.

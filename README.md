# Accelerated Subset-Sum Problem Solver: CPU/GPU/Cooperative Implementation

## Project Overview

This project implements and evaluates multiple computational approaches to solving the **Subset-Sum Problem (SSP)**, a classic NP-Complete problem in computer science. The implementation is based on the algorithmic principles described in the paper *"A novel cooperative accelerated parallel two-list algorithm for solving the subset-sum problem on a hybrid CPU-GPU cluster"* (Wan et al., 2016).

#### Our Kaggle Link : 
https://www.kaggle.com/code/shreyaanbanerjee/notebook92d6c3a872

### Problem Definition

**Subset-Sum Problem (SSP)**: Given a set of integers $W = \{w_1, w_2, ..., w_n\}$ and a target value $M$, determine if there exists a subset of $W$ that sums exactly to $M$.

This is an NP-Complete problem with exponential time complexity $O(2^n)$ for brute-force approaches.

## Algorithm: Two-List Meet-in-the-Middle (Horowitz-Sahni)

To achieve better performance than brute-force, we implement the **Horowitz-Sahni algorithm**:

### Complexity
- **Time**: $O(2^{n/2})$
- **Space**: $O(2^{n/2})$

### Steps
1. **Partition**: Split the input weights $W$ into two equal halves: $W_1$ and $W_2$
2. **Generation**: Generate all possible subset sums for each half:
   - List A: All subset sums from $W_1$ (size $2^{n/2}$)
   - List B: All subset sums from $W_2$ (size $2^{n/2}$)
   - Each entry stores `{sum, bitmask}` to recover the original subset indices
3. **Sort**: Sort List A in ascending order
4. **Search**: For each element $b$ in List B, use binary search to find $a$ in List A such that $a + b = M$

## Implementation Modes

We have implemented **four distinct computational modes** to compare performance:

### 1. Sequential (Baseline)
- Pure Python/NumPy implementation
- Single-threaded execution
- Used as the baseline for speedup calculations

### 2. CPU Parallel
- PyTorch on CPU
- Leverages multi-threaded BLAS/OpenMP backends
- Utilizes all available CPU cores

### 3. GPU
- PyTorch CUDA implementation
- All operations (generation, sorting, binary search) performed on GPU
- Optimized for Tesla T4 GPU (16GB VRAM)

### 4. Cooperative (Hybrid CPU-GPU)
- **List A**: Generated on CPU
- **List B**: Generated on GPU (concurrently with CPU)
- **Search**: Performed on GPU or CPU depending on memory availability
- **Optimizations**:
  - Pinned memory for faster CPU→GPU transfers
  - Non-blocking asynchronous transfers
  - Intelligent device selection heuristic

## Technical Implementation Details

### Core Components

#### `SubsetSumSolver` Class
The main solver class implementing all four modes:

**Key Methods**:
- `_generate_list()`: Vectorized subset sum generation using PyTorch
- `_search()`: Binary search using `torch.searchsorted()`
- `solve_cpu()`: CPU-only execution
- `solve_gpu()`: GPU-only execution
- `solve_cooperative()`: Hybrid CPU-GPU execution

#### Memory Safety
- `check_memory_requirements()`: Pre-execution memory estimation
- Prevents Out-Of-Memory (OOM) crashes
- Checks both system RAM and GPU VRAM
- Uses 80% safety margin

#### Instance Generation
- `generate_instance()`: Creates random SSP instances
- 50% probability of guaranteed solution
- Configurable bit-length for weight values

### Cooperative Mode Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  COOPERATIVE MODE                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐              ┌──────────────┐        │
│  │   CPU Task   │              │   GPU Task   │        │
│  │  (Thread 1)  │              │  (Main)      │        │
│  ├──────────────┤              ├──────────────┤        │
│  │ Generate     │  Concurrent  │ Generate     │        │
│  │ List A       │ ◄──────────► │ List B       │        │
│  │ (W1 subset)  │   Execution  │ (W2 subset)  │        │
│  └──────┬───────┘              └──────┬───────┘        │
│         │                             │                 │
│         │  Pinned Memory              │                 │
│         └─────────────┬───────────────┘                 │
│                       ▼                                  │
│              ┌─────────────────┐                        │
│              │ Memory Heuristic│                        │
│              │  Decision       │                        │
│              └────────┬────────┘                        │
│                       │                                  │
│         ┌─────────────┴─────────────┐                   │
│         ▼                           ▼                    │
│  ┌─────────────┐            ┌─────────────┐            │
│  │ GPU Search  │            │ CPU Search  │            │
│  │ (if memory  │            │ (fallback)  │            │
│  │  available) │            │             │            │
│  └─────────────┘            └─────────────┘            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Experimental Setup

### Hardware Configuration
- **CPU**: 2 logical cores (detected via `psutil`)
- **RAM**: System memory (detected dynamically)
- **GPU**: NVIDIA Tesla T4 (16GB VRAM)
- **Framework**: PyTorch 2.9.0+cu126

### Test Parameters
- **Problem Sizes**: $N \in \{42, 44, 46, 48\}$
- **Instances per Size**: 100
- **Weight Bit-Length**: 40 bits
- **Solution Probability**: 50%

### Why These Problem Sizes?
- $N=42$: List size = $2^{21}$ ≈ 2M elements ≈ 32 MB per list
- $N=44$: List size = $2^{22}$ ≈ 4M elements ≈ 64 MB per list
- $N=46$: List size = $2^{23}$ ≈ 8M elements ≈ 128 MB per list
- $N=48$: List size = $2^{24}$ ≈ 16M elements ≈ 256 MB per list

Larger values ($N \geq 50$) risk OOM errors even on 16GB GPU.

## Performance Metrics

### Primary Metrics

#### 1. Execution Time
- **Definition**: Wall-clock time from start to completion of solve operation
- **Unit**: Seconds
- **Measurement**: Python `time.time()` before/after solver call

#### 2. Speedup
- **Definition**: Ratio of baseline time to accelerated time
- **Formula**: $\text{Speedup} = \frac{T_{\text{CPU}}}{T_{\text{mode}}}$
- **Baseline**: CPU-only median execution time per problem size
- **Interpretation**: Higher is better (e.g., 20x = 20 times faster than CPU)

#### 3. Improvement Percentage
- **Cooperative vs CPU**: $\frac{T_{\text{CPU}} - T_{\text{Coop}}}{T_{\text{CPU}}} \times 100\%$
- **Cooperative vs GPU**: $\frac{T_{\text{GPU}} - T_{\text{Coop}}}{T_{\text{GPU}}} \times 100\%$

### Secondary Metrics

#### 4. Memory Utilization
- **GPU Memory Allocated**: `torch.cuda.memory_allocated()`
- **System RAM Available**: `psutil.virtual_memory().available`
- **Transfer Size**: Bytes transferred CPU↔GPU

#### 5. Success Rate
- **Definition**: Percentage of instances solved without OOM/Error
- **Formula**: $\frac{\text{Successful Runs}}{\text{Total Runs}} \times 100\%$

## Results Summary

### Execution Time Comparison (Median, N=48)

| Mode         | Time (s) | Speedup vs CPU |
|--------------|----------|----------------|
| CPU          | ~12.00   | 1.0x           |
| GPU          | ~0.17    | ~70x           |
| Cooperative  | ~0.61    | ~19x           |

### Key Findings

1. **GPU Mode is Fastest**: Pure GPU consistently outperforms all other modes
2. **Cooperative Mode Bottleneck**: CPU generation phase limits overall performance
3. **Scalability**: GPU time scales better than Cooperative mode as $N$ increases
4. **Memory Constraints**: All modes successfully handled $N \leq 48$ without OOM

### Speedup Trends

```
N=42: GPU ~40x, Cooperative ~20x
N=44: GPU ~44x, Cooperative ~20x
N=46: GPU ~61x, Cooperative ~19x
N=48: GPU ~70x, Cooperative ~19x
```

**Observation**: GPU speedup increases with problem size, while Cooperative speedup plateaus.

## Optimizations Implemented

### Cooperative Mode Enhancements

1. **Pinned Memory**
   ```python
   cpu_result['sums'] = cpu_result['sums'].pin_memory()
   cpu_result['masks'] = cpu_result['masks'].pin_memory()
   ```
   - Enables faster DMA transfers to GPU
   - Reduces CPU→GPU transfer latency

2. **Non-Blocking Transfers**
   ```python
   A_sums_gpu = cpu_result['sums'].to(device_gpu, non_blocking=True)
   torch.cuda.current_stream().synchronize()
   ```
   - Asynchronous data movement
   - CPU can continue while transfer is in progress

3. **Intelligent Device Selection**
   - Heuristic checks available GPU memory
   - Moves CPU data to GPU if space available (fast search)
   - Falls back to CPU search if GPU memory insufficient

4. **Memory Management**
   - `torch.cuda.empty_cache()` after operations
   - Explicit cleanup of intermediate tensors
   - Prevents memory fragmentation

## File Structure

```
HonProj/
├── notebook92d6c3a872.ipynb    # Main implementation notebook
├── Analysis_Report.md           # Detailed performance analysis
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── update_notebook.py           # Utility: Update notebook cells
├── remove_redundant_imports.py # Utility: Code cleanup
├── LanjunWan.pdf               # Reference paper
└── execution_time_comparison.png # Performance visualization
```

## Dependencies

```
torch>=2.0.0
numpy
pandas
matplotlib
seaborn
psutil
```

Install via:
```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

1. Open `notebook92d6c3a872.ipynb` in Jupyter
2. Execute cells sequentially
3. Modify experiment parameters in `run_experiment()` call:
   ```python
   exp_data = run_experiment(
       n_values=[42, 44, 46, 48],  # Problem sizes
       num_instances=100            # Runs per size
   )
   ```

### Standalone Solver Usage

```python
from notebook92d6c3a872 import SubsetSumSolver

# Create instance
weights = [1, 5, 10, 25, 50]
target = 60
solver = SubsetSumSolver(weights, target)

# Solve using different modes
solution_cpu = solver.solve_cpu()
solution_gpu = solver.solve_gpu()
solution_coop = solver.solve_cooperative()

# solution is a list of indices or None
if solution_cpu:
    print(f"Solution indices: {solution_cpu}")
    print(f"Sum: {sum(weights[i] for i in solution_cpu)}")
```

## Comparison with Literature

### Original Paper Context
The Wan et al. (2016) paper targets **distributed MPI clusters** with:
- Multiple CPU nodes
- Limited GPU resources per node
- Inter-node communication overhead

### Our Implementation Context
Single-machine environment with:
- 1 CPU (2 cores)
- 1 GPU (Tesla T4)
- PCIe 3.0 interconnect

### Why Cooperative Mode is Slower
1. **Compute Disparity**: GPU generates its list ~100x faster than CPU
2. **Idle Time**: GPU waits for CPU to finish before search phase
3. **Transfer Overhead**: Moving 256MB over PCIe takes non-trivial time

### When Cooperative Mode Excels
- **Large $N$ ($\geq 60$)**: When lists exceed GPU memory
- **Cluster Environments**: When aggregate CPU power matches GPU
- **Memory-Constrained Scenarios**: When GPU VRAM is limited

## Future Work

### Potential Optimizations
1. **Asymmetric Workload Distribution**: Generate 10% on CPU, 90% on GPU
   - Challenge: Meet-in-the-middle requires balanced split
2. **Multi-GPU Support**: Distribute List B across multiple GPUs
3. **Distributed Computing**: Adapt for MPI cluster (as in original paper)
4. **CPU Optimization**: Use AVX-512 intrinsics for vectorization

### Scalability Analysis
- **$N=50$**: ~1GB per list (feasible on Tesla T4)
- **$N=60$**: ~16GB per list (requires distributed memory)
- **$N=70$**: ~256GB per list (requires cluster)

## References

1. Wan, L., et al. (2016). "A novel cooperative accelerated parallel two-list algorithm for solving the subset-sum problem on a hybrid CPU-GPU cluster." *Journal of Parallel and Distributed Computing*.

2. Horowitz, E., & Sahni, S. (1974). "Computing partitions with applications to the knapsack problem." *Journal of the ACM*, 21(2), 277-292.

## License

This project is for educational and research purposes.

## Authors

- Implementation: Samar Mittal
- Analysis: Conducted as part of Honours Project
- Date: December 2025

## Contact

For questions or collaboration, please refer to the project repository.

---

**Note**: This implementation demonstrates the trade-offs between different computational approaches to NP-Complete problems and highlights the importance of matching algorithmic design to hardware capabilities.

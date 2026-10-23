# Assignment 2: Optimised Matrix Multiplication and Sparse Representations

Empirical study evaluating cache-friendly dense optimisations and compressed sparse row (CSR) representations in Java.

## 1. System and Hardware Environment
- **CPU:** Intel Core i7 (Document physical/logical cores)
- **RAM:** 16 GB DDR4/DDR5
- **Operating System:** Windows 11 / WSL
- **Runtime Environment:** Java 21 LTS (Oracle HotSpot 64-Bit Server VM)

## 2. Experimental Protocol
- **Part A (Dense Optimisation):** Comparison between naive $i\text{-}j\text{-}k$ baseline, loop-reordered $i\text{-}k\text{-}j$, and cache-blocked ($32 \times 32$ tile) kernels. Tested across $n \in \{100, 200, 400, 600, 800, 1000\}$ with 5 measured repetitions following JIT warm-up.
- **Part B (Sparse Representations):** Custom Compressed Sparse Row (CSR) implementation evaluated against dense $i\text{-}k\text{-}j$ across declared sparsity levels ($50\%, 80\%, 90\%, 95\%, 98\%, 99\%$) for $n=500$.
- **Validation:** Results verified against the naive baseline with a numerical tolerance of $\varepsilon = 10^{-9}$.

## 3. Build and Execution Instructions

### Compilation
```bash
javac MatrixOpt.java
javac MatrixSparse.java
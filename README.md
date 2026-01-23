

<h1 align="center"> linghe </h1>

<div style="text-align: center;">
<img src="assets/linghe.png" alt="Logo" width="200">
</div>
  
<p align="center">
   A library of high-performance kernels for LLM training.
</p>



## Roadmap ##
---

- Support more shapes and various GPU archs.
- Release our fp8 training kernels beyond blockwise quantization.

## *News or Update* 🔥
---
- [2025/07] We implement multiple kernels for FP8 training with `Megatron-LM` blockwise quantization. 


## Introduction
---
Our repo, linghe, is designed for LLM training, especially for MoE training with FP8 quantizaiton. It provides 3 main categories of kernels:

- **Fused quantization kernels**: fuse quantization with previous layer, e.g., RMS norm and Silu.
- **Memory-efficiency kernels**: fuse multiple IO-itensive operations, e.g., ROPE with qk-norm.
- **Implementation-optimized kernels**: use efficient triton implementation, e.g., routing map padding instead of activation padding.


## Benchmark
---
We benchmark on H800 with batch size 8192, hidden size 2048, num experts 256, activation experts 8.

| kernel | baseline(us) | linghe(us) | speedup |
|--------|--------------|------------|---------|
| RMSNorm+Quantization(forward) | 159.3 us | 72.4 us    | 2.2 |
| Split+qk-norm+rope+transpose(forward) | 472 us | 59.1 us    | 7.99 |
| Split+qk-norm+rope+transpose(backward) | 645 us | 107.5 us   | 6.0 |
| Fp32 router gemm(forward) | 242.3 us | 61.6 us    | 3.931 |
| Fp32 router gemm(backward) | 232.7 us | 78.1 us    | 2.979 |
| Permute with padded indices | 388 us | 229.4 us   | 1.69 |
| Unpermute with padding indices | 988.6 us | 806.9 us   | 1.23 |
| Batch Silu+quantization(forward) | 6241.7 us | 1181.7 us  | 5.28 |
| Batch Silu+quantization(backward) | 7147.7 us | 2317.9 us  | 3.08 |
| Silu+quantization(forward) | 144.9 us | 58.2 us    | 2.48 |
| Silu+quantization(backward) | 163.4 us | 74.2 us    | 2.2 |
| fused linear gate(forward) | 160.4 us | 46.9 us    | 3.42 |
| fused linear gate(backward) | 572.9 us | 81.1 us    | 7.06 |
| Cross entropy(forward) | 2780.8 us | 818.2 us   | 3.4 |
| Cross entropy(backward) | 7086.3 us | 1781.0 us  | 3.98 |
| batch grad norm | 1733.7 us | 1413.7 us  | 1.23 | 
| Batch count zero | 4997.9 us | 746.8 us   | 6.69 | 

Other benchmark results can be obtained by running scripts in tests and benchmark folders.

## Examples
---

Examples can be found in tests.

## Api Reference
---

Please refer to [API](https://inclusionai.github.io/linghe/)

## Citations

[TBD]
```
@misc{zhao2025linghe,
title={Linghe: Enabling Efficient Trillion-Scale LLM Training via Optimized Kernels},
author={Yao Zhao and Chen Liang and Jingyu Hu and Zixuan Cheng and Longfei Li}
}
```

# Changelog


## linghe 0.4.1(26.04.4)

- optimize topk kernel


## linghe 0.4.0(26.04.02)

- optimize infer kernels


## linghe 0.3.9(26.03.29)

- optimize infer kernels


## linghe 0.3.8(26.03.27)

- optimize infer kernels


## linghe 0.3.7(26.03.25)

- support native format for attention gate kernel
- optimize infer kernels


## linghe 0.3.6(26.03.24)

- split dim for embedding backward
- tune param for gemm


## linghe 0.3.5(26.03.02)

- fix parameter mismatching in parallel ce loss kernel
- support tp in varlen rope
- refine infer kernels


## linghe 0.3.4

- add tma persistent fp32 gemm kernel
- add multiple kernels for inference


## linghe 0.3.3

- use fix chunk size in loading cu_seqlens in rope and mla kernels
- fix accumulation bug in embedding lookup kernel


## linghe 0.3.2

- add the parameter `transpose` to `group_rms_norm_gate`


## linghe 0.3.0

- use faster and more accurate implementation for `embedding` backward
- add multiple embedding_lookup implementations
- use 2048 block number for `batch_count_zero` kernel


## linghe 0.2.9

- support stride in grad tensor of `embedding` kernel
- return grad for dummy tensor in `embedding` kernel
- support bf16 in batch mul/clip/norm kernels


## linghe 0.2.8

- fix racing condition bug in softmax_cross_entropy kernel
- use tl.rsqrt instead of 1/tl.sqrt in all kernels
- add the parameter `tp_group` to `softmax_cross_entropy`


## linghe 0.2.7

- add the parameter `ignore_index` to `softmax_cross_entropy`
- support parallel `softmax_cross_entropy`
- add dtype and numel assertion in multiple batch kernels

- Known issues:
  - performance of `softmax_cross_entropy` degrades when vocab size is not multiple of 16

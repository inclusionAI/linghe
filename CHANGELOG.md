# Changelog




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

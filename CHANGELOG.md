# Changelog

## linghe 0.2.7

- support arg `ignore_index` in `softmax_cross_entropy`
- support parallel `softmax_cross_entropy`
- add dtype and numel assertion in multiple batch kernels

- Known issues:
  - performance of `softmax_cross_entropy` degrades when vocab size is not multiple of 16

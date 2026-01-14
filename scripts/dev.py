import torch
import triton
import triton.language as tl


def test_cpu_gpu_diff():
    x = torch.randn((128, 128), dtype=torch.float32) * 100
    cos = torch.cos(x)
    c = torch.cos(x.cuda())
    torch.testing.assert_close(cos, c.cpu())


@triton.jit
def index_overflow(x):
    i = tl.program_id(0)
    ptr = x + i * 2 ** 30
    offs = 2 ** 30 + 2 ** 30 + 2 ** 30 + 2 ** 30
    # offs = (i).to(tl.int64) * 2 ** 30 + 5*2**32
    tl.store(x + i, offs)


def test_index_overflow():
    x = torch.zeros((128,), dtype=torch.int64, device='cuda:0')
    index_overflow[(128,)](x)
    print(x)


if __name__ == '__main__':
    # test_cpu_gpu_diff()
    test_index_overflow()

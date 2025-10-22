import numpy as np
import torch
import matplotlib.pyplot as plt



def read_bf16_inputs(prefix='fc2'):
    idx = {"qkv":0, "out":1, "fc1s":2, "fc2s":3, "fc1":4, "fc2":5}[prefix]

    d = torch.load(f'/tmp/deepseek/bf16_forward_{idx}.bin', weights_only=True)
    N, K = d['w'].shape 
    M = d['x'].numel()//K

    x = d['x'].detach().float().view(M,K)
    w = d['w'].data.float().view(N,K)

    d = torch.load(f'/tmp/deepseek/bf16_backward_{idx}.bin', weights_only=True)
    dy = d['dy'].detach().float().view(M,N)
    dx = d['dx'].detach().float().view(M,K)

    d = torch.load(f'/tmp/deepseek/bf16_update_{idx}.bin', weights_only=True)
    dw = d['dw'].detach().float().view(N,K)

    x = x.cuda()
    w = w.cuda().transpose().contiguous()
    dy = dy.cuda()
    dx = dx.cuda()
    dw = dw.cuda().transpose().contiguous()
    return x,w,dy,dx,dw

def read_fp8_inputs(prefix='fc2'):
    idx = {"qkv":0, "out":1, "fc1s":2, "fc2s":3, "fc1":4, "fc2":5}[prefix]
    d = torch.load(f'/tmp/deepseek/fp8_forward_{idx}.bin', weights_only=True)
    N,K = d['w'].shape 
    M = d['x'].numel()//K

    xq = d['x'].float().view(M,K)
    xs = d['xs'].float()
    xm = d['x_smooth_scale']
    x = xq*xm*xs[:,None]

    wq = d['w'].float()
    ws = d['ws'].float()
    wm = d['w_smooth_scale']
    w = wq*wm*ws[:,None]

    d = torch.load(f'/tmp/deepseek/fp8_backward_{idx}.bin', weights_only=True)
    dyq = d['dy'].float()
    dys = d['dys']
    dym = d['dy_smooth_scale']
    dy = dyq/dym*dys[:,None]

    dytq = d['dyt'].float().t()
    dyts = d['dyts']
    dytm=d['dyt_smooth_scale']
    dyt = dytq/dytm*dyts[:,None]

    x = x.cuda()
    w = w.cuda().transpose().contiguous()
    dy = dy.cuda()
    dyt = dyt.cuda()
    xm = xm.cuda()
    wm = wm.cuda()
    return x,w,dy,dyt,xm,wm


# bf16
if True:
    prefix = 'out'
    x,w,dy,dx,dw = read_bf16_inputs(prefix=prefix)

    r = 5
    xb = torch.nn.functional.max_pool2d(x.abs()[None],r).cpu().numpy()[0]
    wb = torch.nn.functional.max_pool2d(w.abs()[None],r).cpu().numpy()[0]
    dyb = torch.nn.functional.max_pool2d(dy.abs()[None],r).cpu().numpy()[0]
    dxb = torch.nn.functional.max_pool2d(dx.abs()[None],r).cpu().numpy()[0]
    dwb = torch.nn.functional.max_pool2d(dw.abs()[None],r).cpu().numpy()[0]

    fmt = 'png'
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(xb, cmap='gray')
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/{prefix}_x.{fmt}", bbox_inches='tight',dpi=600)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(wb, cmap='gray')
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/{prefix}_w.{fmt}", bbox_inches='tight',dpi=600)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(dyb, cmap='gray')
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/{prefix}_dy.{fmt}", bbox_inches='tight', dpi=600)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(dxb, cmap='gray')
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/{prefix}_dx.{fmt}", bbox_inches='tight', dpi=600)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(dwb, cmap='gray')
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/{prefix}_dw.{fmt}", bbox_inches='tight', dpi=600)
    plt.close('all')

# fp8
if False:
    prefix = 'fc2'
    x,w,dy,dyt,xm,wm = read_fp8_inputs(prefix=prefix)

    r = 5
    xb = torch.nn.functional.max_pool2d(x.abs()[None],r).cpu().numpy()[0]
    wb = torch.nn.functional.max_pool2d(w.abs()[None],r).cpu().numpy()[0]
    dyb = torch.nn.functional.max_pool2d(dy.abs()[None],r).cpu().numpy()[0]

    fmt = 'png'
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(xb, cmap='gray')
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/{prefix}_x.{fmt}", bbox_inches='tight',dpi=600)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(wb, cmap='gray')
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/{prefix}_w.{fmt}", bbox_inches='tight',dpi=600)
    plt.close('all')

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(dyb, cmap='gray')
    # plt.show()
    plt.axis('off')
    plt.savefig(f"figures/{prefix}_dy.{fmt}", bbox_inches='tight', dpi=600)
    plt.close('all')




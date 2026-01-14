# -*- coding: utf-8 -*-
"""
Copyright (c) Ant Financial Service Group and its affiliates.
"""

import math

import torch


def output_check(org_out, opt_out, name='', rtol=None, atol=None, itol=0,
                 amp=1.0, digest=4):
    org_out = org_out.detach()
    opt_out = opt_out.detach()
    assert org_out.dtype == opt_out.dtype, f"ref:{org_out.dtype} != out:{opt_out.dtype}"
    assert org_out.shape == opt_out.shape, f"ref:{org_out.shape} != out:{opt_out.shape}"
    if org_out.numel() == 0:
        return
    org_dtype = org_out.dtype
    opt_dtype = opt_out.dtype

    if org_dtype in (
    torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2):
        org_out = org_out.float()
    elif org_dtype in (
    torch.bool, torch.uint8, torch.int8, torch.uint16, torch.int16):
        org_out = org_out.int()

    if opt_dtype in (
    torch.bfloat16, torch.float16, torch.float8_e4m3fn, torch.float8_e5m2):
        opt_out = opt_out.float()
    elif org_dtype in (
    torch.bool, torch.uint8, torch.int8, torch.uint16, torch.int16):
        opt_out = opt_out.int()

    if rtol is None:
        if org_dtype == torch.float64:
            rtol = 1e-8
        elif org_dtype == torch.float32:
            rtol = 1e-4
        elif org_dtype == torch.float16:
            rtol = 4e-3
        elif org_dtype == torch.bfloat16:
            rtol = 2e-2
        elif org_dtype == torch.float8_e4m3fn:
            rtol = 0.125
        elif org_dtype == torch.float8_e5m2:
            rtol = 0.25

    if atol is None:
        if org_dtype == torch.float64:
            atol = 1e-12
        elif org_dtype == torch.float32:
            atol = 1e-6
        elif org_dtype == torch.float16:
            atol = 1e-4
        elif org_dtype == torch.bfloat16:
            atol = 1e-3
        elif org_dtype == torch.float8_e4m3fn:
            atol = 0.125
        elif org_dtype == torch.float8_e5m2:
            atol = 0.25

    if org_out.dtype in (torch.float32, torch.float64):
        rtol = rtol * amp
        atol = atol * amp
        diff = (opt_out - org_out).abs()
        abs_error = diff.mean().item()
        rel_error = abs_error / max(org_out.abs().mean().item(), 1e-30)
        if rel_error >= 0.005:
            rel_err_str = f"\033[91m {rel_error:.6f}\033[00m"
        else:
            rel_err_str = f"{rel_error:.6f}"
        org_max = org_out.abs().max()
        org_mean = org_out.abs().mean()
        opt_max = opt_out.abs().max()
        opt_mean = opt_out.abs().mean()
        print(f'\n{name:<16}  rel:{rel_err_str}  abs:{abs_error:.6f}  ' \
              f'org:{org_max:.3f}/{org_mean:.3f} ' \
              f'opt:{opt_max:.3f}/{opt_mean:.3f} ')
        if (rtol >= 0 and atol >= 0):
            # torch.testing.assert_close(opt_out, org_out, rtol=rtol, atol=atol)
            mistake_mask = diff >= (rtol * org_out.abs() + atol)
            if mistake_mask.float().sum().item() > 0:
                org_val = org_out[mistake_mask]
                opt_val = opt_out[mistake_mask]
                mismatch_count = org_val.numel()
                tot_cnt = org_out.numel()
                itv = max(mismatch_count // digest, 1)
                org_val = org_val[::itv].tolist()
                opt_val = opt_val[::itv].tolist()
                if org_dtype == torch.float64:
                    org_str = ', '.join([f'{x:.8g}' for x in org_val])
                    opt_str = ', '.join([f'{x:.8g}' for x in opt_val])
                elif org_dtype == torch.float32:
                    org_str = ', '.join([f'{x:.5g}' for x in org_val])
                    opt_str = ', '.join([f'{x:.5g}' for x in opt_val])
                else:
                    org_str = ', '.join([f'{x:.3g}' for x in org_val])
                    opt_str = ', '.join([f'{x:.3g}' for x in opt_val])
                info = f"Mismatched elements: {mismatch_count} / {tot_cnt} ({mismatch_count / tot_cnt * 100:.1f}%) " \
                       f"with {rtol} rtol and {atol} atol \n        org: {org_str} \n        opt: {opt_str} \n"
                assert mismatch_count == 0, info
        return rel_error
    else:
        # int dtype
        diff = (opt_out - org_out).abs()
        mismatch_count = (diff > itol).sum().item()
        if mismatch_count > 0:
            diff_err_str = f"\033[91m {mismatch_count}\033[00m"
        else:
            diff_err_str = f"{mismatch_count}"
        max_error = diff.max()
        print(f'\n{name:<16}  diff:{diff_err_str} max:{max_error}')
        assert mismatch_count == 0, f"Mismatched elements: {mismatch_count} with {itol} itol"
        return mismatch_count


def quant_check(org_out, xq, wq, opt_out, mode):
    abs_error = (opt_out.float() - org_out.float()).abs().mean().item()
    rel_error = abs_error / org_out.float().abs().mean().item()
    x_underflow = (xq == 0.0).sum().item() / xq.numel()
    w_underflow = (wq == 0.0).sum().item() / wq.numel()
    x_overflow = (torch.isnan(xq)).sum().item()
    w_overflow = (torch.isnan(wq)).sum().item()
    print(f'\n{mode}  rel:{rel_error:.3f}  abs:{abs_error:.3f}  ' \
          f'org:{org_out.abs().max():.3f}/{org_out.abs().mean():.3f} ' \
          f'opt:{opt_out.abs().max():.3f}/{opt_out.abs().mean():.3f} ' \
          f'x_underflow:{x_underflow:.5f} w_underflow:{w_underflow:.5f} ' \
          f'x_overflow:{x_overflow} w_overflow:{w_overflow}')


def inf_or_nan(xs, name=''):
    if not isinstance(xs, (list, tuple)):
        xs = [xs]
    hit = False
    for x in xs:
        value = x.abs().max().item()
        if math.isnan(value) or math.isinf(value):
            hit = True
            break
    if hit:
        for x in xs:
            print(
                f'{name=} {x.shape=} {x.argmax()=} {x.max()=} {x.argmin()=}  {x.min()=} {x=}')

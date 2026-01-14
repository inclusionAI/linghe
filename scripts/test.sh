
cd tests &&
# echo "test_add.py" && python test_add.py && 
# echo "test_blockwise_fp8_gemm.py" && python test_blockwise_fp8_gemm.py && 
# echo "test_blockwise_quant.py" && python test_blockwise_quant.py && 
# echo "test_channel_quant.py" && python test_channel_quant.py && 
# echo "test_channelwise_fp8_gemm.py" && python test_channelwise_fp8_gemm.py && 
# echo "test_embedding.py" && python test_embedding.py  && 
# echo "test_fp32_gemm.py" && python test_fp32_gemm.py && 
echo "test_gate.py" && python test_gate.py && 
echo "test_gather.py" && python test_gather.py  && 
echo "test_group_quant.py" && python test_group_quant.py  && 
echo "test_hadamard_quant.py" && python test_hadamard_quant.py  && 
# echo "test_la.py" && python test_la.py && 
echo "test_loss.py" && python test_loss.py && 
# echo "test_mla.py" && python test_mla.py && 
echo "test_mul.py" && python test_mul.py  && 
echo "test_norm.py" && python test_norm.py  && 
echo "test_rearange.py" && python test_rearange.py  && 
echo "test_reduce.py" && python test_reduce.py  && 
echo "test_rope.py" && python test_rope.py  && 
echo "test_scatter.py" && python test_scatter.py  && 
echo "test_silu.py" && python test_silu.py  && 
echo "test_smooth_quant.py" && python test_smooth_quant.py  && 
echo "test_topk.py" && python test_topk.py  && 
echo "test_transpose.py" && python test_transpose.py && 
echo "test_unary.py" && python test_unary.py && 
echo "success!"


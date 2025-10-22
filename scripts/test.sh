
cd ../tests &&
echo "test_add.py" && python test_add.py 
echo "test_channel_quant.py" && python test_channel_quant.py
echo "test_dot.py" && python test_dot.py 
echo "test_gather.py" && python test_gather.py 
echo "test_gemm.py" && python test_gemm.py
echo "test_group_quant.py" && python test_group_quant.py 
echo "test_loss.py" && python test_loss.py 
echo "test_norm.py" && python test_norm.py 
echo "test_rearange.py" && python test_rearange.py 
echo "test_reduce.py" && python test_reduce.py 
echo "test_scatter.py" && python test_scatter.py 
echo "test_silu.py" && python test_silu.py 
echo "test_smooth_quant.py" && python test_smooth_quant.py 
echo "test_transpose.py" && python test_transpose.py
echo "test_unary.py" && python test_unary.py


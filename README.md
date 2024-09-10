# kernel:

- laynorm (9/4)
	-- v0: base
	-- v1: block_reduce (normal)
	-- v2: block_reduce (warp reduce)
	
- softmax (9/5)
	-- v0: base
	-- v1: block_reduce (normal)
	-- v2: block_reduce (warp reduce)

- matrixMul
	-- v0: base （9/7）
	-- v1: 共享内存分块 (9/7)
	-- v2: 寄存器分块
	-- v3: 重排索引
	-- v4: float4 访存
	-- v5: 解 bank 冲突
	-- v6: 内积转外积
	-- v7: 双缓冲
	

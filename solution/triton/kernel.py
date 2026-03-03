import math
import torch
import triton
import triton.language as tl

@triton.jit
def _moe_le_fused_kernel(
    # Hidden states and scales
    hs_ptr, hs_scale_ptr,
    T, H, I,
    # Token routing
    sorted_tokens_ptr, w_tok_ptr,
    # Block mapping
    b_expert_id_ptr, b_token_offset_ptr, b_num_tokens_ptr,
    # Expert weights and scales
    w13_ptr, s13_ptr,
    w2_ptr, s2_ptr,
    # Workspace for intermediate activations C
    workspace_ptr,
    # Output
    out_ptr,
    # Strides
    stride_hs_t, stride_hs_h,
    stride_hss_hb, stride_hss_t,
    stride_w13_e, stride_w13_o, stride_w13_h,
    stride_s13_e, stride_s13_ob, stride_s13_hb,
    stride_w2_e, stride_w2_h, stride_w2_i,
    stride_s2_e, stride_s2_hb, stride_s2_ib,
    stride_out_t, stride_out_h,
    # Constants
    NUM_H_BLOCKS: tl.constexpr,   # 56
    NUM_I_BLOCKS: tl.constexpr,   # 16
    BLOCK_M: tl.constexpr,        # 64
    BLOCK_K: tl.constexpr,        # 128
    BLOCK_I: tl.constexpr,        # 128
    BLOCK_N: tl.constexpr         # 128
):
    pid = tl.program_id(0)
    
    # Block details
    expert_id = tl.load(b_expert_id_ptr + pid)
    token_offset = tl.load(b_token_offset_ptr + pid)
    num_tokens = tl.load(b_num_tokens_ptr + pid)
    
    # Token indices [BLOCK_M]
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = offs_m < num_tokens
    
    tok_idx = tl.load(sorted_tokens_ptr + token_offset + offs_m, mask=mask_m, other=0)
    weight = tl.load(w_tok_ptr + token_offset + offs_m, mask=mask_m, other=0.0)
    
    # -----------------------------------------------------------
    # GEMM1 & SwiGLU: Loop over I chunks
    # -----------------------------------------------------------
    for ib in range(NUM_I_BLOCKS):
        u1 = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
        u2 = tl.zeros((BLOCK_M, BLOCK_I), dtype=tl.float32)
        
        offs_i = ib * BLOCK_I + tl.arange(0, BLOCK_I)
        
        for kb in range(NUM_H_BLOCKS):
            offs_k = kb * BLOCK_K + tl.arange(0, BLOCK_K)
            
            # Load A tile [BLOCK_M, BLOCK_K]
            a_ptrs = hs_ptr + tok_idx[:, None] * stride_hs_t + offs_k[None, :] * stride_hs_h
            a_fp8 = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)  # Native FP8
            
            # Load A scale [1, BLOCK_M]
            sA = tl.load(hs_scale_ptr + kb * stride_hss_hb + tok_idx * stride_hss_t, mask=mask_m, other=0.0)
            
            # Load W1 (Gate) [BLOCK_I, BLOCK_K]
            w1_ptrs = w13_ptr + expert_id * stride_w13_e + offs_i[:, None] * stride_w13_o + offs_k[None, :] * stride_w13_h
            w1_fp8 = tl.load(w1_ptrs)  # Native FP8
            sW1 = tl.load(s13_ptr + expert_id * stride_s13_e + ib * stride_s13_ob + kb * stride_s13_hb)
            
            # Load W3 (Up) [BLOCK_I, BLOCK_K]
            offs_i_up = I + offs_i
            w3_ptrs = w13_ptr + expert_id * stride_w13_e + offs_i_up[:, None] * stride_w13_o + offs_k[None, :] * stride_w13_h
            w3_fp8 = tl.load(w3_ptrs)  # Native FP8
            sW3 = tl.load(s13_ptr + expert_id * stride_s13_e + (NUM_I_BLOCKS + ib) * stride_s13_ob + kb * stride_s13_hb)
            
            # Native FP8 @ FP8 dot product -> returned as FP32
            raw_dot_1 = tl.dot(a_fp8, tl.trans(w1_fp8), out_dtype=tl.float32)
            raw_dot_2 = tl.dot(a_fp8, tl.trans(w3_fp8), out_dtype=tl.float32)
            
            # Scale and accumulate
            u1 += raw_dot_1 * (sA[:, None] * sW1)
            u2 += raw_dot_2 * (sA[:, None] * sW3)
            
        # Apply SwiGLU
        silu = u2 / (1.0 + tl.exp(-u2))
        c = silu * u1
        
        # Store to workspace: [total_routes, I]
        c_ptrs = workspace_ptr + (token_offset + offs_m)[:, None] * I + offs_i[None, :]
        tl.store(c_ptrs, c, mask=mask_m[:, None])
        
    tl.debug_barrier()
    
    # -----------------------------------------------------------
    # GEMM2: Loop over H chunks
    # -----------------------------------------------------------
    for nb in range(NUM_H_BLOCKS):
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        o_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for ib in range(NUM_I_BLOCKS):
            offs_i = ib * BLOCK_I + tl.arange(0, BLOCK_I)
            
            # Load C_chunk from workspace [BLOCK_M, BLOCK_I]
            c_ptrs = workspace_ptr + (token_offset + offs_m)[:, None] * I + offs_i[None, :]
            c_chunk = tl.load(c_ptrs, mask=mask_m[:, None], other=0.0)
            c_f32 = c_chunk
            
            # Load W2 tile [BLOCK_N, BLOCK_I]
            w2_ptrs = w2_ptr + expert_id * stride_w2_e + offs_n[:, None] * stride_w2_h + offs_i[None, :] * stride_w2_i
            w2_fp8 = tl.load(w2_ptrs)
            
            # Load W2 scale
            sW2 = tl.load(s2_ptr + expert_id * stride_s2_e + nb * stride_s2_hb + ib * stride_s2_ib)
            
            # Scale W2 and cast to FP16 to use FP16 Tensor Cores
            w2_scaled = w2_fp8.to(tl.float32) * sW2
            
            # Mixed FP16/BF16 dot depending on hardware -> accumulate to FP32
            o_acc += tl.dot(c_f32, tl.trans(w2_scaled), out_dtype=tl.float32)
            
        # Multiply by routing weight
        o_acc = o_acc * weight[:, None]
        # Write to final output with atomic_add in FP32
        out_ptrs = out_ptr + tok_idx[:, None] * stride_out_t + offs_n[None, :] * stride_out_h
        tl.atomic_add(out_ptrs, o_acc, mask=mask_m[:, None])

def _check_cuda_and_move(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device.type == 'cuda':
        return t
    if device.type != 'cuda':
        raise RuntimeError("CUDA is required to run this kernel; no CUDA device available.")
    return t.to(device, non_blocking=True)

def _ensure_cuda(*tensors):
    if not torch.cuda.is_available():
        for t in tensors:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                raise RuntimeError("CUDA inputs provided but CUDA is reported unavailable.")
        raise RuntimeError("CUDA is required to run this kernel; no CUDA device available.")
    return torch.device('cuda')

@torch.no_grad()
def run(
    routing_logits: torch.Tensor,
    routing_bias: torch.Tensor,
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    local_expert_offset: int,
    routed_scaling_factor: float,
):
    H = 7168
    I = 2048
    E_global = 256
    E_local = 32
    TOP_K = 8
    N_GROUP = 8
    TOPK_GROUP = 4
    BLOCK = 128
    NUM_H_BLOCKS = H // BLOCK            # 56
    NUM_I_BLOCKS = I // BLOCK            # 16
    NUM_G1_BLOCKS = (2 * I) // BLOCK     # 32

    T = int(routing_logits.shape[0])

    device = _ensure_cuda(routing_logits, routing_bias, hidden_states, hidden_states_scale,
                          gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale)
    orig_device = routing_logits.device

    # Move tensors to CUDA contiguous
    routing_logits_cu = _check_cuda_and_move(routing_logits, device).contiguous()
    routing_bias_cu = _check_cuda_and_move(routing_bias.to(torch.float32), device).contiguous()
    hidden_states_cu = _check_cuda_and_move(hidden_states, device).contiguous()
    hidden_states_scale_cu = _check_cuda_and_move(hidden_states_scale, device).contiguous()
    gemm1_weights_cu = _check_cuda_and_move(gemm1_weights, device).contiguous()
    gemm1_weights_scale_cu = _check_cuda_and_move(gemm1_weights_scale, device).contiguous()
    gemm2_weights_cu = _check_cuda_and_move(gemm2_weights, device).contiguous()
    gemm2_weights_scale_cu = _check_cuda_and_move(gemm2_weights_scale, device).contiguous()

    # 1) DeepSeek-V3 no-aux routing mapped to PyTorch
    logits = routing_logits_cu.to(torch.float32)                      
    bias = routing_bias_cu.to(torch.float32).reshape(-1)              
    s = torch.sigmoid(logits)                                         
    s_with_bias = s + bias                                            

    group_size = E_global // N_GROUP  
    s_wb_grouped = s_with_bias.view(T, N_GROUP, group_size)           
    top2_vals, _ = torch.topk(s_wb_grouped, k=2, dim=2, largest=True, sorted=False)  
    group_scores = top2_vals.sum(dim=2)                               
    _, group_idx = torch.topk(group_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)

    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = group_mask.unsqueeze(2).expand(T, N_GROUP, group_size).reshape(T, E_global)

    neg_inf = torch.finfo(torch.float32).min
    scores_pruned = s_with_bias.masked_fill(score_mask == 0, neg_inf)
    _, topk_idx = torch.topk(scores_pruned, k=TOP_K, dim=1, largest=True, sorted=False)

    M = torch.zeros_like(s)
    M.scatter_(1, topk_idx, 1.0)
    weights = s * M
    weights_sum = weights.sum(dim=1, keepdim=True) + 1e-20
    weights = (weights / weights_sum) * float(routed_scaling_factor)

    # 2) Token flattening and expert sorting
    local_start = int(local_expert_offset)
    local_end = local_start + E_local
    
    valid_mask = (topk_idx >= local_start) & (topk_idx < local_end)
    token_idx_flat = torch.arange(T, device=device).unsqueeze(1).expand(-1, TOP_K)[valid_mask]
    expert_idx_flat = topk_idx[valid_mask] - local_start
    
    if token_idx_flat.numel() == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=orig_device)
        
    sorted_indices = torch.argsort(expert_idx_flat)
    sorted_tokens = token_idx_flat[sorted_indices].to(torch.int32).contiguous()
    sorted_experts = expert_idx_flat[sorted_indices].to(torch.int32).contiguous()
    
    w_tok = weights[sorted_tokens.to(torch.int64), sorted_experts.to(torch.int64) + local_start].to(torch.float32).contiguous()
    
    # 3) Block mappings for kernel
    BLOCK_M = 64
    BLOCK_K = 128
    BLOCK_I = 128
    BLOCK_N = 128

    expert_counts = torch.bincount(sorted_experts, minlength=E_local)
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int32, device=device)
    expert_offsets[1:] = torch.cumsum(expert_counts, dim=0)
    
    expert_counts_cpu = expert_counts.cpu().tolist()
    expert_offsets_cpu = expert_offsets.cpu().tolist()
    
    block_expert_id = []
    block_token_offset = []
    block_num_tokens = []
    
    for le_id in range(E_local):
        count = expert_counts_cpu[le_id]
        if count == 0:
            continue
        start_off = expert_offsets_cpu[le_id]
        num_blocks = (count + BLOCK_M - 1) // BLOCK_M
        for b in range(num_blocks):
            block_expert_id.append(le_id)
            block_token_offset.append(start_off + b * BLOCK_M)
            chunk = min(BLOCK_M, count - b * BLOCK_M)
            block_num_tokens.append(chunk)
            
    total_blocks = len(block_expert_id)
    if total_blocks == 0:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=orig_device)
        
    b_expert_id = torch.tensor(block_expert_id, dtype=torch.int32, device=device)
    b_token_offset = torch.tensor(block_token_offset, dtype=torch.int32, device=device)
    b_num_tokens = torch.tensor(block_num_tokens, dtype=torch.int32, device=device)

    # 4) Launch massively fused kernel
    workspace = torch.empty((sorted_tokens.size(0), I), dtype=torch.float32, device=device)
    out_accum = torch.zeros((T, H), dtype=torch.float32, device=device)
    
    grid = (total_blocks,)

    _moe_le_fused_kernel[grid](
        hidden_states_cu, hidden_states_scale_cu,
        T, H, I,
        sorted_tokens, w_tok,
        b_expert_id, b_token_offset, b_num_tokens,
        gemm1_weights_cu, gemm1_weights_scale_cu,
        gemm2_weights_cu, gemm2_weights_scale_cu,
        workspace,
        out_accum,
        # Strides
        hidden_states_cu.stride(0), hidden_states_cu.stride(1),
        hidden_states_scale_cu.stride(0), hidden_states_scale_cu.stride(1),
        gemm1_weights_cu.stride(0), gemm1_weights_cu.stride(1), gemm1_weights_cu.stride(2),
        gemm1_weights_scale_cu.stride(0), gemm1_weights_scale_cu.stride(1), gemm1_weights_scale_cu.stride(2),
        gemm2_weights_cu.stride(0), gemm2_weights_cu.stride(1), gemm2_weights_cu.stride(2),
        gemm2_weights_scale_cu.stride(0), gemm2_weights_scale_cu.stride(1), gemm2_weights_scale_cu.stride(2),
        out_accum.stride(0), out_accum.stride(1),
        # Consts
        NUM_H_BLOCKS, NUM_I_BLOCKS, BLOCK_M, BLOCK_K, BLOCK_I, BLOCK_N,
        num_warps=8,
        num_stages=3
    )

    if orig_device.type != 'cuda':
        out_accum = out_accum.cpu()

    return out_accum.to(torch.bfloat16)

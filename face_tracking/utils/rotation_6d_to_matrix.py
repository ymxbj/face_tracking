import torch
import torch.nn.functional as F
import triton
import triton.language as tl

# --- Triton 前向核函数 (修改后) ---
@triton.jit
def _rotation_6d_to_matrix_forward_kernel(
    d6_ptr,           # 输入: [*, 6]
    rot_mat_ptr,      # 输出: [*, 3, 3]
    num_elements,     # 总共需要转换的向量数 (*)
    EPSILON: tl.constexpr, 
    BLOCK_SIZE: tl.constexpr, 
):
    """
    Triton kernel for converting 6D rotation representation to a 3x3 rotation matrix.
    Each program instance handles one 6D vector.
    """
    # 获取当前 program 的 ID
    # 这里我们不再使用 pid = tl.program_id(axis=0) 的简单方式
    # 而是采用更通用的、按块处理的方式
    pid_block_start = tl.program_id(axis=0) * BLOCK_SIZE
    offsets = pid_block_start + tl.arange(0, BLOCK_SIZE)
    
    # 创建一个 mask 来防止越界访问
    # 当 num_elements 不是 BLOCK_SIZE 的整数倍时，这非常重要
    mask = offsets < num_elements
    
    # --- 1. 加载 a1 和 a2 ---
    # 使用 tl.load 和 mask 来安全地加载数据
    # tl.load(pointer, mask=mask, other=0.0) 表示如果 mask 为 False，就加载 other 指定的默认值
    a1x = tl.load(d6_ptr + offsets * 6 + 0, mask=mask)
    a1y = tl.load(d6_ptr + offsets * 6 + 1, mask=mask)
    a1z = tl.load(d6_ptr + offsets * 6 + 2, mask=mask)
    a2x = tl.load(d6_ptr + offsets * 6 + 3, mask=mask)
    a2y = tl.load(d6_ptr + offsets * 6 + 4, mask=mask)
    a2z = tl.load(d6_ptr + offsets * 6 + 5, mask=mask)

    
    # --- 2. 计算 b1 = normalize(a1) ---
    norm_a1 = tl.sqrt(a1x * a1x + a1y * a1y + a1z * a1z + EPSILON)
    b1x = a1x / norm_a1
    b1y = a1y / norm_a1
    b1z = a1z / norm_a1

    # --- 3. 计算 b2 ---
    dot_b1_a2 = b1x * a2x + b1y * a2y + b1z * a2z
    b2_un_x = a2x - dot_b1_a2 * b1x
    b2_un_y = a2y - dot_b1_a2 * b1y
    b2_un_z = a2z - dot_b1_a2 * b1z
    norm_b2_un = tl.sqrt(b2_un_x * b2_un_x + b2_un_y * b2_un_y + b2_un_z * b2_un_z + EPSILON)
    b2x = b2_un_x / norm_b2_un
    b2y = b2_un_y / norm_b2_un
    b2z = b2_un_z / norm_b2_un

    # --- 4. 计算 b3 = cross(b1, b2) ---
    b3x = b1y * b2z - b1z * b2y
    b3y = b1z * b2x - b1x * b2z
    b3z = b1x * b2y - b1y * b2x

    # --- 5. 将 b1, b2, b3 写入输出矩阵 ---
    # 使用 tl.store 和 mask 安全地写入
    tl.store(rot_mat_ptr + offsets * 9 + 0, b1x, mask=mask)
    tl.store(rot_mat_ptr + offsets * 9 + 1, b1y, mask=mask)
    tl.store(rot_mat_ptr + offsets * 9 + 2, b1z, mask=mask)
    
    tl.store(rot_mat_ptr + offsets * 9 + 3, b2x, mask=mask)
    tl.store(rot_mat_ptr + offsets * 9 + 4, b2y, mask=mask)
    tl.store(rot_mat_ptr + offsets * 9 + 5, b2z, mask=mask)
    
    tl.store(rot_mat_ptr + offsets * 9 + 6, b3x, mask=mask)
    tl.store(rot_mat_ptr + offsets * 9 + 7, b3y, mask=mask)
    tl.store(rot_mat_ptr + offsets * 9 + 8, b3z, mask=mask)


# 将原始的 PyTorch 函数重命名，以便在 backward 中调用
def rotation_6d_to_matrix_pytorch(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


class Rotation6DToMatrixTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, d6: torch.Tensor):
        # 保存原始形状，并将输入展平为 [N, 6]
        original_shape = d6.shape
        d6_reshaped = d6.reshape(-1, 6).contiguous()
        
        # 检查输入
        assert d6_reshaped.is_cuda and d6_reshaped.is_contiguous(), "Input tensor must be a contiguous CUDA tensor"
        
        num_elements = d6_reshaped.shape[0]
        
        # 创建输出张量
        rot_mat_reshaped = torch.empty(num_elements, 3, 3, device=d6.device, dtype=d6.dtype)
        
        # 设置 Triton Grid
        # 每个 program 处理一个 6D 向量
        grid = lambda meta: (triton.cdiv(num_elements, meta['BLOCK_SIZE']),)
        
        # 启动 Triton 核函数
        _rotation_6d_to_matrix_forward_kernel[grid](
            d6_reshaped,
            rot_mat_reshaped,
            num_elements,
            EPSILON=1e-8,
            BLOCK_SIZE=1024 # 可以调整的块大小
        )
        
        # 保存输入以用于反向传播
        ctx.save_for_backward(d6)
        
        # 将输出恢复到原始批处理形状
        return rot_mat_reshaped.reshape(*original_shape[:-1], 3, 3)

    @staticmethod
    def backward(ctx, grad_rot_mat: torch.Tensor):
        """
        Calculates the gradient for the 6D rotation conversion.
        
        This method uses a hybrid approach:
        1. It re-enables gradient computation locally using `with torch.enable_grad():`
           to handle cases where the backward pass is called within a no-grad context
           (e.g., during evaluation loops `with torch.no_grad():`).
        2. It reconstructs the computation graph by re-running the original,
           differentiable PyTorch function.
        3. It leverages `torch.autograd.grad` to automatically and correctly compute
           the Vector-Jacobian Product, avoiding complex and error-prone manual
           gradient derivation.
        """
        # 恢复前向传播时保存的输入张量
        d6, = ctx.saved_tensors

        with torch.enable_grad():
            d6_with_grad = d6.detach().requires_grad_(True)
            
            # 使用原始的 PyTorch 函数重新执行前向传播，构建计算图
            rot_mat_pytorch = rotation_6d_to_matrix_pytorch(d6_with_grad)
        
        # 使用 torch.autograd.grad 自动计算梯度
        grad_d6 = torch.autograd.grad(
            outputs=rot_mat_pytorch,
            inputs=d6_with_grad,
            grad_outputs=grad_rot_mat,
            create_graph=torch.is_grad_enabled(),
        )[0]
        
        return grad_d6
import torch
import triton
import triton.language as tl

# 确保安装了 triton
# !pip install triton

# ------------------------------------
# 1. Triton 前向核函数
# ------------------------------------
@triton.jit
def vertices2joints_forward_kernel(
    # --- 输入张量 ---
    vertices_ptr,          # [B, V, 3] 顶点数据
    sparse_weights_ptr,    # [NNZ] J_regressor的非零权重
    sparse_verts_idxes_ptr,# [NNZ] 顶点索引
    sparse_joint_idxes_ptr,# [NNZ] 关节索引
    # --- 输出张量 ---
    joints_ptr,            # [B, J, 3] 关节数据 (输出)
    # 步长 (stride)
    v_stride_b, v_stride_v,
    j_stride_b, j_stride_j,
    # --- Triton 元参数 ---
):
    """
    Triton Kernel for forward pass: joints = J_regressor @ vertices
    Grid: (NNZ, B)
    每个 program 计算一个非零元素对一个batch item的贡献。
    """
    # 获取当前 program 的 ID
    pid_nnz = tl.program_id(axis=0)  # 当前处理第几个非零元素 (0 to NNZ-1)
    pid_batch = tl.program_id(axis=1) # 当前处理第几个 batch (0 to B-1)

    # 1. 加载稀疏矩阵信息 (权重、顶点索引、关节索引)
    weight = tl.load(sparse_weights_ptr + pid_nnz)
    vert_idx = tl.load(sparse_verts_idxes_ptr + pid_nnz)
    joint_idx = tl.load(sparse_joint_idxes_ptr + pid_nnz)

    # 2. 加载对应的顶点坐标 (x, y, z)
    # 计算顶点数据的内存地址
    # offset for vertices[pid_batch, vert_idx, :]
    v_offset = pid_batch * v_stride_b + vert_idx * v_stride_v
    # 因为 D=3 是个很小的值，直接分3次加载比用 mask 更简单高效
    vx = tl.load(vertices_ptr + v_offset + 0)
    vy = tl.load(vertices_ptr + v_offset + 1)
    vz = tl.load(vertices_ptr + v_offset + 2)

    # 3. 计算加权后的顶点坐标
    weighted_vx = vx * weight
    weighted_vy = vy * weight
    weighted_vz = vz * weight

    # 4. 使用 atomic_add 将结果累加到输出的关节位置
    # 计算输出关节数据的内存地址
    # offset for joints[pid_batch, joint_idx, :]
    j_offset = pid_batch * j_stride_b + joint_idx * j_stride_j
    
    # 原子操作：将加权后的坐标累加到对应的关节上
    # 多个非零元素可能贡献给同一个关节，因此必须用原子操作防止写入冲突
    tl.atomic_add(joints_ptr + j_offset + 0, weighted_vx)
    tl.atomic_add(joints_ptr + j_offset + 1, weighted_vy)
    tl.atomic_add(joints_ptr + j_offset + 2, weighted_vz)


# ------------------------------------
# 2. Triton 反向核函数
# ------------------------------------
@triton.jit
def vertices2joints_backward_kernel(
    # --- 输入张量 ---
    grad_joints_ptr,       # [B, J, 3] 上游传来的梯度
    sparse_weights_ptr,    # [NNZ] J_regressor的非零权重
    sparse_verts_idxes_ptr,# [NNZ] 顶点索引
    sparse_joint_idxes_ptr,# [NNZ] 关节索引
    # --- 输出张量 ---
    grad_vertices_ptr,     # [B, V, 3] 顶点梯度 (输出)
    # 步长
    gj_stride_b, gj_stride_j,
    gv_stride_b, gv_stride_v,
):
    """
    Triton Kernel for backward pass: grad_vertices = J_regressor.T @ grad_joints
    Grid: (NNZ, B)
    """
    pid_nnz = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)

    # 1. 加载稀疏矩阵信息
    weight = tl.load(sparse_weights_ptr + pid_nnz)
    vert_idx = tl.load(sparse_verts_idxes_ptr + pid_nnz)
    joint_idx = tl.load(sparse_joint_idxes_ptr + pid_nnz)
    
    # 2. 加载对应的上游梯度 grad_joints
    # offset for grad_joints[pid_batch, joint_idx, :]
    gj_offset = pid_batch * gj_stride_b + joint_idx * gj_stride_j
    grad_jx = tl.load(grad_joints_ptr + gj_offset + 0)
    grad_jy = tl.load(grad_joints_ptr + gj_offset + 1)
    grad_jz = tl.load(grad_joints_ptr + gj_offset + 2)

    # 3. 计算对顶点的梯度贡献
    # grad_v = grad_j * weight
    grad_vx_contrib = grad_jx * weight
    grad_vy_contrib = grad_jy * weight
    grad_vz_contrib = grad_jz * weight

    # 4. 使用 atomic_add 将梯度贡献累加到对应的顶点梯度上
    # offset for grad_vertices[pid_batch, vert_idx, :]
    gv_offset = pid_batch * gv_stride_b + vert_idx * gv_stride_v
    
    # 原子操作：一个顶点可能被多个(J_regressor.T的)非零元素影响
    tl.atomic_add(grad_vertices_ptr + gv_offset + 0, grad_vx_contrib)
    tl.atomic_add(grad_vertices_ptr + gv_offset + 1, grad_vy_contrib)
    tl.atomic_add(grad_vertices_ptr + gv_offset + 2, grad_vz_contrib)

class Vertices2JointsTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vertices, sparse_weights, sparse_verts_idxes, sparse_joint_idxes, num_joints):
        # --- 输入校验 ---
        assert vertices.is_cuda and vertices.is_contiguous(), "Input 'vertices' must be a contiguous CUDA tensor"
        assert sparse_weights.is_cuda and sparse_weights.is_contiguous(), "Input 'sparse_weights' must be a contiguous CUDA tensor"
        assert sparse_verts_idxes.is_cuda and sparse_verts_idxes.is_contiguous(), "Input 'sparse_verts_idxes' must be a contiguous CUDA tensor"
        assert sparse_joint_idxes.is_cuda and sparse_joint_idxes.is_contiguous(), "Input 'sparse_joint_idxes' must be a contiguous CUDA tensor"
        
        # --- 获取维度 ---
        batch_size, num_vertices, D = vertices.shape
        nnz = sparse_weights.shape[0]
        assert D == 3, "Vertex dimension must be 3"

        # --- 创建输出张量 ---
        joints = torch.zeros(batch_size, num_joints, D, device=vertices.device, dtype=vertices.dtype)

        # --- 设置Triton Kernel的Grid ---
        # 我们为每个(非零元素, batch)组合启动一个program
        grid = (nnz, batch_size)

        # --- 启动前向Kernel ---
        vertices2joints_forward_kernel[grid](
            vertices, sparse_weights, sparse_verts_idxes, sparse_joint_idxes,
            joints, vertices.stride(0), vertices.stride(1),
            joints.stride(0), joints.stride(1),
        )

        # --- 保存反向传播所需的张量 ---
        ctx.save_for_backward(sparse_weights, sparse_verts_idxes, sparse_joint_idxes)
        ctx.v_shape = vertices.shape
        
        return joints

    @staticmethod
    def backward(ctx, grad_joints):
        # --- 输入校验 ---
        grad_joints = grad_joints.contiguous()
        assert grad_joints.is_cuda, "Input 'grad_joints' must be a CUDA tensor"

        # --- 恢复前向传播保存的张量和信息 ---
        sparse_weights, sparse_verts_idxes, sparse_joint_idxes = ctx.saved_tensors
        batch_size, num_vertices, D = ctx.v_shape
        num_joints = grad_joints.shape[1]
        nnz = sparse_weights.shape[0]

        # --- 创建输出梯度张量 ---
        grad_vertices = torch.zeros(batch_size, num_vertices, D, device=grad_joints.device, dtype=grad_joints.dtype)

        # --- 设置Triton Kernel的Grid ---
        grid = (nnz, batch_size)

        # --- 启动反向Kernel ---
        vertices2joints_backward_kernel[grid](
            grad_joints, sparse_weights, sparse_verts_idxes, sparse_joint_idxes,
            grad_vertices, grad_joints.stride(0), grad_joints.stride(1),
            grad_vertices.stride(0), grad_vertices.stride(1),
        )
        
        # backward函数需要为forward的每个输入返回一个梯度
        # sparse_weights, idxes等不需要梯度，返回None
        return grad_vertices, None, None, None, None

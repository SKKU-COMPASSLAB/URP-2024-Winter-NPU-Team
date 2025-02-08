# -*- coding: utf-8 -*-
import torch
import functools
import numpy as np
from typing import Union
import threading
import queue
from queue import Queue
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import torchvision
import torchvision.transforms as transforms
from torch.nn import Conv2d, Linear
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import requests
from typing import Tuple
from typing import List




def cast2any8type(arr: torch.Tensor, dst_dtype: torch.dtype):
    if not isinstance(arr, torch.Tensor):
        raise Exception(f"[ERROR] the 'arr' should be torch.Tensor, not {type(arr).__name__}")
    if dst_dtype == torch.uint8:
        np_arr = arr.contiguous().cpu().numpy()
        np_bytes = np_arr.view(np.uint8)
        return torch.from_numpy(np_bytes.copy()).flatten()
    else:
        return arr.flatten().clone().to(dst_dtype)

class SystolicArrayOS:
    def __init__(self, arr_height: int, arr_width: int, dtype: torch.dtype, onchip_mem_size: int=102400*102400):
        self.arr_height = arr_height
        self.arr_width = arr_width
        self.dtype = dtype
        self.onchip_mem_size = onchip_mem_size
        
        self.itemsize = torch.tensor([], dtype=dtype).element_size()
        


        self.onchip_mem = torch.zeros(onchip_mem_size, dtype=torch.uint8)
        # For an output-stationary design, partial sums live in acc_registers across calls.
        self.acc_registers = torch.zeros((self.arr_height, self.arr_width), dtype=self.dtype)

    def check_onchip_mem_access(self, addr: int, size: int) -> bool:
        return (addr >= 0) and (size > 0) and (addr + size <= self.onchip_mem_size)

    def read_onchip_mem(self, addr: int, shape: Union[torch.Size, tuple]):
        size = functools.reduce(lambda a, b: a * b, shape, 1) * self.itemsize
        if not self.check_onchip_mem_access(addr, size):
            raise Exception(
                f"[ERROR] onchip memory request is invalid (addr={addr}, "
                f"size={size}, capacity={self.onchip_mem_size})"
            )
        raw_bytes = self.onchip_mem[addr : addr + size].clone()
        np_bytes = raw_bytes.cpu().numpy()
        # dtype 변환 시 float16/float32 등 여러 경우를 고려해야 하지만
        # 여기서는 torch.float로 고정 가정(또는 torch.finfo(self.dtype).dtype).
        np_restored = np_bytes.view(np.dtype(torch.finfo(self.dtype).dtype))
        return torch.from_numpy(np_restored.copy()).reshape(shape)

    def Aload(self, addr: int, tensor: torch.Tensor) -> int:
        size = tensor.numel() * tensor.element_size()
        row, column = tensor.size()
        if not self.check_onchip_mem_access(addr, size):
            raise Exception(
                f"[ERROR] onchip memory request is invalid (addr={addr}, "
                f"size={size}, capacity={self.onchip_mem_size})"
            )
        as_bytes = cast2any8type(tensor, torch.uint8)
        self.onchip_mem[addr : addr + size] = as_bytes
        return row
    
    def Bload(self, addr: int, tensor: torch.Tensor) -> int:
        size = tensor.numel() * tensor.element_size()
        row, column = tensor.size()
        if not self.check_onchip_mem_access(addr, size):
            raise Exception(
                f"[ERROR] onchip memory request is invalid (addr={addr}, "
                f"size={size}, capacity={self.onchip_mem_size})"
            )
        as_bytes = cast2any8type(tensor, torch.uint8)
        self.onchip_mem[addr : addr + size] = as_bytes
        return column
    


    def execute(self, a_addr: int, b_addr: int, seq_len: int, rowA: int, colB: int, i_nnz: int) -> int:
        """
        - rowA: 실제 subA의 행 개수(타일 높이)
        - seq_len: A의 열 (K)
        - colB: 실제 subB의 열 개수(타일 폭)
        """
        TPE_r = 8
        TPE_c = 4
        # 실제 shape
        a_shape = (rowA, seq_len)    # (ex: 10 x 80)
        b_shape = (seq_len, colB)    # (ex: 80 x 12)
        self.delay = 0

        a_size = self.arr_height * seq_len * self.itemsize
        b_size = seq_len * self.arr_width * self.itemsize

        if not self.check_onchip_mem_access(a_addr, a_size):
            raise Exception(f"[ERROR] onchip memory request is invalid (addr={a_addr}, size={a_size})")
        if not self.check_onchip_mem_access(b_addr, b_size):
            raise Exception(f"[ERROR] onchip memory request is invalid (addr={b_addr}, size={b_size})")


        A = self.read_onchip_mem(a_addr, a_shape)
        B = self.read_onchip_mem(b_addr, b_shape)

        # (rowA x colB) matmul
        matmul_result = torch.matmul(A, B)   # shape (rowA, colB)

        # acc_registers는 (arr_height, arr_width) = (15, 15),
        # 부분적으로 업데이트: 0..rowA, 0..colB 만 더해준다
        self.acc_registers[:rowA, :colB] += matmul_result
        delay = i_nnz*(10 + round(seq_len/8))
        #print (delay)

        # cycle 계산(예시로 seq_len)
        return delay



    def flush(self, addr: int) -> int:
        """
        Write partial sums to memory, then zero out self.acc_registers.
        """
        size = self.arr_height * self.arr_width * self.itemsize
        if not self.check_onchip_mem_access(addr, size):
            raise Exception(
                f"[ERROR] onchip memory request is invalid (addr={addr}, "
                f"size={size}, capacity={self.onchip_mem_size})"
            )

        as_bytes = cast2any8type(self.acc_registers, torch.uint8)
        self.onchip_mem[addr : addr + size] = as_bytes

        # Clear accumulators
        self.acc_registers = torch.zeros((self.arr_height, self.arr_width), dtype=self.dtype)
        return self.arr_width

def store_tile_in_output(output_tensor: torch.Tensor, tile: torch.Tensor, 
                         row_start: int, col_start: int, valid_h: int, valid_w: int):
    """
    Copy only the valid region (valid_h x valid_w) from top-left of tile
    into output_tensor at (row_start, col_start).
    """
    output_tensor[row_start : row_start + valid_h, 
                  col_start : col_start + valid_w] = tile[:valid_h, :valid_w]

def prune_1d_topk(block_1d: torch.Tensor, nnz: int):
    """
    block_1d에서 절댓값 기준으로 상위 nnz개만 남기고 나머지는 0으로 만든다.
    """
    length = block_1d.numel()
    if nnz <= 0:
        return torch.zeros_like(block_1d)
    abs_vals = block_1d.abs()
    _, top_indices = torch.topk(abs_vals, k=nnz, largest=True)
    pruned = torch.zeros_like(block_1d)
    pruned[top_indices] = block_1d[top_indices]
    return pruned

def prune_input_matrix(input_mat: torch.Tensor, block_size=8, nnz=4):
    """
    (M x K) 행렬을 행(row) 단위로 반복:
      - 한 행을 길이 K
      - 이 행을 block_size만큼 분할 (열 방향)
      - 각 블록에 대해:
         * 비율 ratio = nnz / block_size
         * sub_block 길이에 맞게 nnz_effective = round(ratio * sub_block_size)
         * top-k(prune_1d_topk) 수행
      - pruned_mat에 반영
    """
    M, K = input_mat.shape
    pruned_mat = torch.zeros_like(input_mat)

    ratio = nnz / float(block_size)  # 목표 압축 비율

    for i in range(M):
        row_data = input_mat[i, :]
        for start in range(0, K, block_size):
            end = min(start + block_size, K)
            sub_block = row_data[start:end]
            sub_len = sub_block.numel()

            # 이 sub_block 크기에 대응되는 nnz_effective = round(ratio * sub_len)
            nnz_effective = int(round(ratio * sub_len))
            # nnz_effective는 sub_len보다 클 수 없으므로 min 처리
            nnz_effective = min(nnz_effective, sub_len)
            # nnz_effective가 0보다 작아지지 않도록 max 처리(혹은 0이면 0 그대로)
            nnz_effective = max(nnz_effective, 0)

            pruned_sub = prune_1d_topk(sub_block, nnz_effective)
            pruned_mat[i, start:end] = pruned_sub

    return pruned_mat

def prune_filter_matrix(filter_mat: torch.Tensor, block_size=8, nnz=4):
    """
    (K x N) 행렬을 '행' 단위로 반복:
      - 한 행은 길이 N
      - 이 행을 block_size만큼 분할(열 방향)
      - 비율로 nnz_effective 계산 후 prune
    """
    K, N = filter_mat.shape
    pruned_mat = torch.zeros_like(filter_mat)

    ratio = nnz / float(block_size)

    for i in range(N):
        col_data = filter_mat[:, i]
        for start in range(0, K, block_size):
            end = min(start + block_size, K)
            sub_block = col_data[start:end]
            sub_len = sub_block.numel()

            nnz_effective = int(round(ratio * sub_len))
            nnz_effective = min(nnz_effective, sub_len)
            nnz_effective = max(nnz_effective, 0)

            pruned_sub = prune_1d_topk(sub_block, nnz_effective)
            pruned_mat[start:end, i] = pruned_sub

    return pruned_mat

import torch

def matrix_accuracy_relative(A: torch.Tensor, B: torch.Tensor, eps: float = 1e-10) -> float:
    """
    두 행렬 A, B의 NMAE (Normalized Mean Absolute Error)를 계산하여 반환.
    
    NMAE = mean(|A - B|) / (mean(A) + eps)

    - A.mean()이 0일 경우 eps 추가하여 0 나눗셈 방지.
    """
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A{A.shape}, B{B.shape}")

    # float 변환
    A = A.float()
    B = B.float()

    # NMAE 계산 (eps 추가)
    nmae = (A - B).abs() / (A.abs() + eps)
    nmae = nmae.mean()
    return nmae.item()

def save_to_csv(file_name, headers, data):
    """Save data to a CSV file in append mode."""
    file_exists = os.path.isfile(file_name)  # 파일 존재 여부 확인
    with open(file_name, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:  # 파일이 없으면 헤더 추가
            writer.writerow(headers)
        writer.writerow(data)

def get_last_iteration(file_name):
    """Return the last iteration number from the CSV file, or 0 if the file does not exist."""
    if not os.path.isfile(file_name):
        return 0
    with open(file_name, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        iterations = [int(row["Iteration"]) for row in reader]
        return max(iterations, default=0)
    
def plot_csv_columns(csv_file, x_column, y_column):

    #주어진 CSV 파일에서 특정 두 열을 x축과 y축으로 사용하여 그래프를 그리는 함수.

    # CSV 파일 읽기
    data = pd.read_csv(csv_file)

    # 데이터가 있는지 확인
    if x_column not in data.columns or y_column not in data.columns:
        raise ValueError(f"'{x_column}' 또는 '{y_column}' 열이 CSV 파일에 없습니다.")
    
    # x_column 기준으로 오름차순 정렬
    data = data.sort_values(by=x_column)

    # x축과 y축 값 추출
    x_values = data[x_column]
    y_values = data[y_column]

    # 그래프 그리기
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x_values, y_values, marker='o', linestyle='', color='b', label=f'{y_column} vs {x_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    ax.set_title(f'{y_column} vs {x_column} 그래프')
    ax.legend()
    ax.grid()

def save_plot_to_pdf(fig, pdf_file):
    # 주어진 Figure 객체를 PDF로 저장하는 함수.
    # pdf_file (str): 저장할 PDF 파일 경로
    
    try:
        with PdfPages(pdf_file) as pdf:
            pdf.savefig(fig)
            print(f"그래프가 {pdf_file}에 추가되었습니다.")
    except Exception as e:
        print(f"PDF 저장 중 오류가 발생했습니다: {e}")

def ifm_lowering(tensor: torch.Tensor, weight_shape: Tuple[int], padding: int, stride: int):
    # Input feature map을 GEMM 형식으로 변환
    N, C, H, W = tensor.shape
    _, _, FW, FH = weight_shape # GEMM으로 변환하는 데는 필터의 크기만 필요함. 채널 수, 개수 무시
    
    OH = (H - FH + (2 * padding)) // stride + 1  # output height
    OW = (W - FW + (2 * padding)) // stride + 1  # output width

    # 패딩이 설정된 경우 입력에 패딩 추가
    # 배치와 채널에는 패딩 x, 입력의 크기에만 패딩 추가.
    if padding > 0:
        tensor = torch.nn.functional.pad(tensor, (padding, padding, padding, padding, 0, 0, 0, 0), 'constant', value=0)
        
    tensor = tensor.permute((0, 2, 3, 1))  # N, H, W, C
    output_tensor = torch.zeros(size=(N, OH, OW, FH * FW * C), dtype=tensor.dtype)

    for n in range(N):
        for oh in range(OH):
            for ow in range(OW):
                h = oh * stride
                w = ow * stride
                output_tensor[n, oh, ow, :] = tensor[n, h:h+FH, w:w+FW, :].flatten()

    # 윈도우로 구성된 input map으로 바뀐 행렬을 return
    return output_tensor.reshape((N*OH*OW, FH*FW*C))

def wgt_lowering(tensor: torch.Tensor):
    # 각 열이 하나의 필터가 되도록 변환
    K, _, _, _ = tensor.shape
    tensor = tensor.permute((0, 2, 3, 1))  # K, FH, FW, C
    return tensor.reshape((K, -1)).T

class ExtractionHookSession(object):
    # verbose가 True이면 결과를 출력
    def __init__(self, verbose: bool=False):
        self.verbose = verbose
        
        self.data: dict[str, torch.Tensor] = {}
        
    def create_hook(self, module_name: str, lowering: bool):
        def __hook(module, ref_input: List[torch.Tensor], ref_output):
            ifm = ref_input[0].detach().clone()
            wgt = module.weight.data.detach().clone()
            
            if lowering:
                ifm = ifm_lowering(ifm, wgt.shape, module.padding[0], module.stride[0])
                wgt = wgt_lowering(wgt)

            self.data[module_name + '.ifm'] = ifm
            self.data[module_name + '.wgt'] = wgt

            if self.verbose:
                print(f"extracted {module_name}.ifm")
                print(f"extracted {module_name}.wgt")
                
        return __hook
        
    def register_hook(self, module: torch.nn.Module, lowering: bool, module_name: str=None):
        if module_name is None:
            module_name = module._get_name()
        
        for submodule_name, submodule in module.named_children():
            full_submodule_name = f"{module_name}.{submodule_name}"
            
            if isinstance(submodule, torch.nn.Conv2d):
                submodule.register_forward_hook(self.create_hook(full_submodule_name, lowering=lowering))
                self.data[full_submodule_name + ".ifm"] = None
                self.data[full_submodule_name + ".wgt"] = None
            elif isinstance(submodule, torch.nn.Linear):
                submodule.register_forward_hook(self.create_hook(full_submodule_name, lowering=False))
                self.data[full_submodule_name + ".ifm"] = None
                self.data[full_submodule_name + ".wgt"] = None
            else:
                self.register_hook(module=submodule, lowering=lowering, module_name=full_submodule_name)
                
    def save_data(self, path: str):
        os.makedirs(path, exist_ok=True)
        
        for tensor_name, tensor in self.data.items():
            torch.save(tensor, os.path.join(path, tensor_name))
            
    def restore_data(self, path: str, module: torch.nn.Module, module_name: str=None):
        # 저장되어 있는 데이터 파일에서 ifm과 wgt를 불러옴
        if module_name is None:
            module_name = module._get_name()
        
        for submodule_name, submodule in module.named_children():
            full_submodule_name = f"{module_name}.{submodule_name}"
            
            if isinstance(submodule, torch.nn.Conv2d) or isinstance(submodule, torch.nn.Linear):
                self.data[full_submodule_name + ".ifm"] = torch.load(os.path.join(path, full_submodule_name + '.ifm'), weights_only=True)
                self.data[full_submodule_name + ".wgt"] = torch.load(os.path.join(path, full_submodule_name + '.wgt'), weights_only=True)
            else:
                self.restore_data(path=path, module=submodule, module_name=full_submodule_name)

    def get_layer_type_by_name(self, model, layer_name):
    # 레이어의 이름으로 Conv2d인지 Linear인지 알아냄
        modules = dict(model.named_modules())
        if layer_name in modules:
            return type(modules[layer_name])
        return None

def simulate_systolic_array(a_: torch.Tensor, b_: torch.Tensor, f_nnz, i_nnz,  csv_file: str = "Resnet_s2ta.csv"):
    # 아래 코드는 전체 매트릭스 곱을 타일 단위로 시뮬레이션하는 예시
    sa = SystolicArrayOS(arr_height=32, arr_width=32, dtype=torch.float32)

    M, K = a_.shape
    K2, N = b_.shape

    headers = ["Iteration", "M", "K", "N", "cycle", "Speedup", "Validation", "activation_nnz/BZ", "filter_nnz/BZ", "block_size", "acc"]

    block_sz = 8
    tile_height = 32
    tile_width  = 32
    timestamp   = 0
    compute_cycle = 0
    original_OS_cycle = 0

    iteration = get_last_iteration(csv_file) + 1

    a = prune_input_matrix(a_, block_size=block_sz, nnz=i_nnz)
    b = prune_filter_matrix(b_, block_size=block_sz, nnz=f_nnz)

    output_matrix = torch.zeros((M, N), dtype=torch.float32)
        
    # We'll define on-chip addresses for A, B, D
    a_addr = 0
    b_addr = 8 * 8 * 40000   # 256 bytes offset (8x8 float)
    d_addr = 8 * 8 * 80000   # 512

    for row_start in range(0, M, tile_height):
        for col_start in range(0, N, tile_width):

            # Reset accumulators for this tile
            sa.acc_registers = torch.zeros((tile_height, tile_width), dtype=torch.float32)

            # This tile's "valid" dimensions might be smaller than 15
            actual_tile_h = min(tile_height, M - row_start)
            actual_tile_w = min(tile_width,  N - col_start)

            # Extract sub-tiles for A and B
            subA = a[row_start : row_start + actual_tile_h, :]
            subB = b[:, col_start: col_start + actual_tile_w]

            # Step 1: load A
            timestamp += sa.Aload(a_addr, subA)
            # Step 2: load B
            timestamp += sa.Bload(b_addr, subB)

            # Step 3: run systolic array (SMT-SA 스타일 execute)
            seq_len = K
            rowA,colA = subA.size()
            rowB,colB = subB.size()
            original_OS_cycle += 64 + K - 2
            compute_cycle += sa.execute(a_addr, b_addr, seq_len, rowA, colB, i_nnz)
            # After finishing the K dimension, flush partial sums
            timestamp += sa.flush(d_addr)

            # Read 15x15 out from d_addr
            full_tile = sa.read_onchip_mem(d_addr, (tile_height, tile_width))

            # Copy only the valid region into output
            store_tile_in_output(output_matrix, full_tile, 
                                row_start, col_start, 
                                actual_tile_h, actual_tile_w)

    timestamp += compute_cycle
    speedup = original_OS_cycle/compute_cycle
    # Compare with reference
    reference = torch.matmul(a_, b_)
    acc = matrix_accuracy_relative(reference, output_matrix)
    validation = "passed" if torch.allclose(reference, output_matrix) else "failed"

    torch.set_printoptions(sci_mode=False)

        
    # CSV에 결과 저장
    save_to_csv(csv_file, headers, [iteration, M, K, N, compute_cycle, speedup, validation, i_nnz/block_sz, f_nnz/block_sz, block_sz, acc])

    #fig_3 = plot_csv_columns('s2ta_results.csv', 'activation_nnz/BZ', 'accuracy')
    #save_plot_to_pdf(fig_1, 's2ta_std_speed_graph.pdf')
    return output_matrix

if __name__ == "__main__":
    # STEP 1: create CNN model
    #   - You can download the CNN model from torchvision
    #   - Model downloaded from Huggingface is also compatible with this example code
    
    import torchvision
    
    weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1

    model = torchvision.models.resnet50(weights=weights)
    print(model)
    exit()
    # STEP 2: create extraction session and register hook to the model
    #   - Use 'lowering' option to convert the Conv2d operator into GEMM (General Matrix Multiplication)
    #   - It is recommended to activate the 'lowering' option to simulate with the systolic array behavioral model
    
    session = ExtractionHookSession(verbose=True)
    session.register_hook(module=model, lowering=True)
    
    
    # STEP 3: inference with the data
    #   - Instead of using the zeros tensor, use the imagenet data
    
    
    #x = torch.zeros((1, 3, 224, 224))    # instead of using the zeros tensor, use the imagenet data

    imagenet_transform = transforms.Compose([
        transforms.Resize(256),  # ResNet은 256x256으로 리사이징 후 크롭
        transforms.CenterCrop(224),  # 224x224 크기로 중앙 크롭
        transforms.ToTensor(),  # Tensor 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
    ])

    imagenet_path = "/data/datasets/ILSVRC2012"

    val_path = os.path.join(imagenet_path, "val")
    pin_class = "n03877845"

    #sample_class = os.listdir(val_path)[0]  # 첫 번째 클래스를 선택
    #print(f"The first class in ImageNet val folder: {sample_class}")
    sample_image_path = os.path.join(val_path, pin_class, os.listdir(os.path.join(val_path, pin_class))[0])

    img = Image.open(sample_image_path).convert("RGB")  # 이미지를 RGB 형식으로 변환
    input_tensor = imagenet_transform(img).unsqueeze(0)  # 배치 차원 추가
    
    model = model.eval()
    model(input_tensor)

    # STEP 4: save and restore dump data (optional)
    #   - You can save the dump data to the file system and restore the dump data to the session without 
    #     incurring inference repeatedly
    #   - Stored dump data may require excessive external storage space
    
    path = os.path.join(os.curdir, "dump_data", model._get_name())
    session.save_data(path=path)
    session.restore_data(path=path, module=model)

    # 블록 별 레이어 매핑
    block_mapping = {
        "conv2x": ["ResNet.layer1.0.conv1", "ResNet.layer1.0.conv2", "ResNet.layer1.1.conv1", "ResNet.layer1.1.conv2"],
        "conv3x": ["ResNet.layer2.0.conv1", "ResNet.layer2.0.conv2", "ResNet.layer2.1.conv1", "ResNet.layer2.1.conv2"],
        "conv4x": ["ResNet.layer3.0.conv1", "ResNet.layer3.0.conv2", "ResNet.layer3.1.conv1", "ResNet.layer3.1.conv2"],
        "conv5x": ["ResNet.layer4.0.conv1", "ResNet.layer4.0.conv2", "ResNet.layer4.1.conv1", "ResNet.layer4.1.conv2"],
    }


    ###### case : conv 중 선택해서 돌리기 ######
    # 특정 블록 선택 (conv2x ~ conv5x)
    selected_block = "conv2x"  # conv2x, conv3x, conv4x, conv5x 중 선택

    # 선택한 블록에 포함된 모든 conv 레이어 리스트 가져오기
    selected_layers = block_mapping[selected_block]

    # Systolic Array 실행
    """for layer_name in selected_layers:
        ifm = session.data[f"{layer_name}.ifm"]
        wgt = session.data[f"{layer_name}.wgt"]

        if ifm is None or wgt is None:
            print(f"[WARNING] {layer_name} 데이터가 존재하지 않습니다.")
            continue

        print(f"Processing {layer_name} in {selected_block}...")

        # Systolic Array 적용
        systolic_output = simulate_systolic_array(ifm, wgt)

        # 결과 출력
        print(f"Layer: {selected_layer.replace('.ifm', '')}")
        print(f"Systolic Array Output Shape: {systolic_output.shape}")

    ###### case : conv만 전부 돌리기 ######
    for block_name, selected_layers in block_mapping.items():
        print(f"\n Processing {block_name}...")

        # Systolic Array 실행
        for layer_name in selected_layers:
            ifm = session.data.get(f"{layer_name}.ifm")
            wgt = session.data.get(f"{layer_name}.wgt")

            if ifm is None or wgt is None:
                print(f"[WARNING] {layer_name} 데이터가 존재하지 않습니다.")
                continue

            print(f"Processing {layer_name} in {block_name}...")

            # Systolic Array 적용
            systolic_output = simulate_systolic_array(ifm, wgt)

            # 결과 출력
            print(f"Layer: {layer_name.replace('.ifm', '')}")
            print(f"Systolic Array Output Shape: {systolic_output.shape}")"""

    ###### case : full로 돌리기 ######
    for f_nnz in range (2, 9, 2):
        for i_nnz in range (2, 9, 2):
            for layer_name in session.data.keys():
                if ".ifm" in layer_name:  # ifm 데이터만 순회

                    ifm = session.data[layer_name]
                    wgt = session.data[layer_name.replace(".ifm", ".wgt")]

                    if "downsample.0" in layer_name:
                        continue

                    if "fc" in layer_name:  # Fully Connected (Linear) 계층이면 Transpose 적용
                        wgt = wgt.t()
                        if ifm.dim() > 2:  
                            ifm = ifm.view(ifm.size(0), -1)
                
                    
                    name = f"resnet_s2ta_f{f_nnz}_i{i_nnz}.csv"

                    # Systolic Array 적용
                    systolic_output = simulate_systolic_array(ifm, wgt, f_nnz, i_nnz, name)

                    # 결과 출력
                    print(f"Layer: {layer_name.replace('.ifm', '')}")
                    print(f"Systolic Array Output Shape: {systolic_output.shape}")

    
    # STEP 5: check the extracted data
    #   - 'session.data' is a dictionary that stores the data
    #   - The name of the tensor is the key of the dictionary and the values are the corresponding tensor
    #   - If the name ends with the '.ifm', then the tensor is an intermediate input feature map (IFM) of the correponding layer
    #   - If the name ends with the '.wgt', then the tensor is a weight parameters
    
    #for tensor_name, tensor in session.data.items():
        #print(f"{tensor_name:30s}: {tensor.shape}")


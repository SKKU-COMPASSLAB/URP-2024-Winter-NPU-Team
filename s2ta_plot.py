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
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def cast2any8type(arr: torch.Tensor, dst_dtype: torch.dtype):
    if not isinstance(arr, torch.Tensor):
        raise Exception(f"[ERROR] the 'arr' should be torch.Tensor, not {type(arr).__name__}")
    if dst_dtype == torch.uint8:
        np_arr = arr.cpu().numpy()
        np_bytes = np_arr.view(np.uint8)
        return torch.from_numpy(np_bytes.copy()).flatten()
    else:
        return arr.flatten().clone().to(dst_dtype)

class SystolicArrayOS:
    def __init__(self, arr_height: int, arr_width: int, dtype: torch.dtype, onchip_mem_size: int=1024*1024):
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
        print (delay)

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

def generate_matrix_with_zeros(rows, cols, zero_percentage=0.75):
    """Generate a matrix with the specified percentage of zeros"""
    # 전체 원소 수와 0의 개수 계산
    total_elements = rows * cols
    num_zeros = int(total_elements * zero_percentage)
    
    # 먼저 모든 원소를 0으로 초기화
    mat = torch.zeros((rows, cols), dtype=torch.float32)
    # 0이 아닌 값을 넣을 위치를 랜덤하게 선택 (중복없이)
    indices = torch.randperm(total_elements)[num_zeros:]  # num_zeros 이후의 인덱스가 non-zero 위치
    # 선택된 위치에 1-100 사이의 랜덤값 삽입
    row_indices = indices // cols
    col_indices = indices % cols
    non_zero_values = torch.randint(1, 9, (len(indices),), dtype=torch.float32)
    mat[row_indices, col_indices] = non_zero_values
    
    # 값 확인을 위한 출력
    num_actual_zeros = (mat == 0).sum().item()
    print(f"\nGenerated matrix with {num_actual_zeros} zeros ({num_actual_zeros/total_elements:.1%})")
    
    return mat

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


    

    # 상대 오차 계산: abs(B - A) / (abs(A) + eps)
    diff = (B - A).abs()
    denom = A.abs() + eps  # A[i]가 0일 때 eps로 대체
    rel_err = diff / denom

    # 평균 상대 오차
    mre = rel_err.mean().item()

    # 정확도 = 1 - 평균 상대 오차
    accuracy = 1.0 - mre
    return accuracy

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


if __name__ == "__main__":
    # 아래 코드는 전체 매트릭스 곱을 타일 단위로 시뮬레이션하는 예시
    sa = SystolicArrayOS(arr_height=32, arr_width=32, dtype=torch.float32)
    M, N, K = 512, 512, 1024
    block_sz = 8
    tile_height = 32
    tile_width  = 32
    timestamp   = 0
    compute_cycle = 0
    i_nnz = 1
    f_nnz = 1
    print(f"\nMatrix sizes:")
    print(f"A: {M}x{K}")
    print(f"B: {K}x{N}")
    print(f"Systolic Array size: {M}x{N}")

    
    
    csv_file = "s2ta_results.csv"
    headers = ["Iteration", "M", "K", "N", "cycle", "Speedup", "accuracy", "activation_nnz/BZ", "filter_nnz/BZ", "block_size"]
    last_iteration = get_last_iteration(csv_file)
    print(last_iteration)


    for z in range(1,9):
        f_nnz = z
        for c in range(1,9):
            i_nnz = c
            for iteration in range(last_iteration + 1, last_iteration + 3):
            
                a_ = generate_matrix_with_zeros(M,K, 0.75)
                b_ = generate_matrix_with_zeros(K,N, 0.375) 
                a = prune_input_matrix(a_, block_size=block_sz, nnz=i_nnz)
                b = prune_filter_matrix(b_, block_size=block_sz, nnz=f_nnz)


                output_matrix = torch.zeros((M, N), dtype=torch.float32)


                # We'll define on-chip addresses for A, B, D
                a_addr = 0
                b_addr = 8 * 8 * 4000   # 256 bytes offset (8x8 float)
                d_addr = 8 * 8 * 8000   # 512
                timestamp   = 0
                compute_cycle = 0
                
                original_OS_cycle = 0
                
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
                        compute_cycle += sa.execute(a_addr, b_addr, seq_len, rowA, colB, i_nnz)
                        # After finishing the K dimension, flush partial sums
                        timestamp += sa.flush(d_addr)

                        # Read 15x15 out from d_addr
                        full_tile = sa.read_onchip_mem(d_addr, (tile_height, tile_width))

                        
                        original_OS_cycle += 64 + K - 2

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
                print(f"simulated computation time: {compute_cycle}")
                print(f"simulated execution time: {timestamp}")
                print(f"result validation: {'passed' if torch.allclose(reference, output_matrix) else 'failed'}")
                print("[INFO] relative-based accuracy =", acc)
                print("reference:")
                print(reference)
                print("simulated:")
                print(output_matrix)

            
                # CSV에 결과 저장
                save_to_csv(csv_file, headers, [iteration, M, K, N, compute_cycle, speedup, acc, i_nnz/block_sz, f_nnz/block_sz, block_sz])

    #fig_1 = plot_csv_columns('s2ta_results.csv', 'activation_nnz/BZ', 'cycle')
    #fig_2 = plot_csv_columns('s2ta_results.csv', 'zero std', 'accuracy')
    fig_3 = plot_csv_columns('s2ta_results.csv', 'activation_nnz/BZ', 'accuracy')
    #save_plot_to_pdf(fig_1, 's2ta_std_speed_graph.pdf')
    #save_plot_to_pdf(fig_2, 's2ta_std_accuracy_graph.pdf')
    save_plot_to_pdf(fig_3, 's2ta_accuracy_graph.pdf')
    print(f"\nResults saved to {csv_file}")

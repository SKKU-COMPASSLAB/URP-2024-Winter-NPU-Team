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
import torch.nn.utils.prune as prune
# -*- coding: utf-8 -*-


###############################
# 1) ArbitrationUnit
###############################
class ArbitrationUnit:
    def __init__(self, word_size=16, parent = None):
        self.word_size = word_size
        self.accumulator_reg = 0
        self.parent = parent
        # per-thread input queues
        self.left_queues = [Queue(maxsize=4) for _ in range(4)]  # four threads
        self.top_queues = [Queue(maxsize=4) for _ in range(4)]   # four threads
        self.left_full_check = [0,0,0,0]
        self.top_full_check = [0,0,0,0]
        self.pointer = 0 #for arbitration unit
        self.right_out = 0
        self.bottom_out = 0
        self.right_out_bypass = []
        self.bottom_out_bypass = []
        self.chosen_thread = -1  # 마지막으로 선택된 thread 기록
        self.bypassed_threads = []
    def arbitration_unit(self, r=None, c=None, check = None):
        bypassed_threads =[]
        # Always define left_vals and top_vals first
        left_vals = [q.queue[0] if not q.empty() else 'p' for q in self.left_queues]
        top_vals  = [q.queue[0] if not q.empty() else 'p' for q in self.top_queues]
        # thread empties
        thread0_empty = self.left_queues[0].empty() or self.top_queues[0].empty()
        thread1_empty = self.left_queues[1].empty() or self.top_queues[1].empty()
        thread2_empty = self.left_queues[2].empty() or self.top_queues[2].empty()
        thread3_empty = self.left_queues[3].empty() or self.top_queues[3].empty()
        thread_empty_list = [thread0_empty,thread1_empty,thread2_empty,thread3_empty]
        pointer_temp = 0
        chosen_thread = -1
        pe_comb = self.parent.mac_array
        
        
        #right는 해당 pe가 오른쪽 pe로만 전파하는 경우
        #bottom은 해당 pe가 아래 pe로만 전파하는 경우
        #all은 오른쪽과 아래 모두 전파하는 경우
        #end는 인접 pe로 전파할 필요 없는 경우
        for i in range(4):#round robin arbitraton
            idx = (self.pointer+i) % 4
            if thread_empty_list[idx] != True:
                if check == "right":
                    if pe_comb[r][c+1].left_queues[idx].full():
                        pass
                    else:
                        if left_vals[idx] == 0 or top_vals[idx] == 0 :
                            bypassed_threads.append(idx)
                        elif chosen_thread == -1:
                            chosen_thread = idx 
                            pointer_temp = (idx+1) % 4 #after thread chosen, the pointer goes to next  
                elif check == "bottom":
                    if pe_comb[r+1][c].top_queues[idx].full():
                        pass
                    else:
                        if left_vals[idx] == 0 or top_vals[idx] == 0 :
                                bypassed_threads.append(idx)
                        elif chosen_thread == -1:
                            chosen_thread = idx 
                            pointer_temp = (idx+1) % 4 #after thread chosen, the pointer goes to next
                elif check == "all":
                    if pe_comb[r+1][c].top_queues[idx].full() or pe_comb[r][c+1].left_queues[idx].full():
                        pass
                    else:
                        if left_vals[idx] == 0 or top_vals[idx] == 0 :
                            bypassed_threads.append(idx)
                        elif chosen_thread == -1:
                            chosen_thread = idx 
                            pointer_temp = (idx+1) % 4 #after thread chosen, the pointer goes to next
                elif check == "end":
                    if left_vals[idx] == 0 or top_vals[idx] == 0 :
                        bypassed_threads.append(idx)
                    elif chosen_thread == -1:
                        chosen_thread = idx 
                        pointer_temp = (idx+1) % 4 #after thread chosen, the pointer goes to next                    
        

        self.pointer = pointer_temp ## pointer update           
              
        a_val, b_val = 0, 0
        
        a_val_bypass_list = []
        b_val_bypass_list = []
        
        if chosen_thread >= 0:
            # dequeue from chosen thread
            if not self.left_queues[chosen_thread].empty():
                a_val = self.left_queues[chosen_thread].get()
            if not self.top_queues[chosen_thread].empty():
                b_val = self.top_queues[chosen_thread].get()
            # MAC operation
            self.accumulator_reg += a_val * b_val
        self.right_out = a_val
        self.bottom_out = b_val
        self.chosen_thread = chosen_thread        
        #for bypass
        for i in bypassed_threads:
            a_val_bypass_list.append(self.left_queues[i].get())
            b_val_bypass_list.append(self.top_queues[i].get())
        
        
        self.right_out_bypass = a_val_bypass_list
        self.bottom_out_bypass = b_val_bypass_list
        self.bypassed_threads = bypassed_threads
    def reset(self):
        # accumulator, current_thread 초기화
        self.accumulator_reg = 0
        self.current_thread = 0

        # 큐 비우기
        for q in self.left_queues:
            while not q.empty():
                q.get()
        for q in self.top_queues:
            while not q.empty():
                q.get()


def status_print(r,c,def_left_queues,def_top_queues,def_chosen_thread,def_bypassed_threads,typing=None):
    print(typing)
    print(f"\nPE({r},{c}) Input Status:")
    print("Thread 0:")
    print(f"  Left Queue: {list(def_left_queues[0].queue)}")
    print(f"  Top Queue: {list(def_top_queues[0].queue)}")
    print("Thread 1:")
    print(f"  Left Queue: {list(def_left_queues[1].queue)}")
    print(f"  Top Queue: {list(def_top_queues[1].queue)}")
    print("Thread 2:")
    print(f"  Left Queue: {list(def_left_queues[2].queue)}")
    print(f"  Top Queue: {list(def_top_queues[2].queue)}")
    print("Thread 3:")
    print(f"  Left Queue: {list(def_left_queues[3].queue)}")
    print(f"  Top Queue: {list(def_top_queues[3].queue)}")
    if typing == "after arbitration":
        print("\nProcessing Results:")      
        print(f"PE({r},{c}) selected thread: {def_chosen_thread}")
        print(f"PE({r},{c}) bypassed thread: {def_bypassed_threads}")      
        
        
        
###############################
# 2) Systolicarray (Verilog -> Python)
###############################
class Systolicarray:


    def __init__(self, ROWS=4, COLS=4, WORD_SIZE=16):
        self.ROWS = ROWS
        self.COLS = COLS
        self.WORD_SIZE = WORD_SIZE

        # 제어 신호
        self.ctl_stat_bit_in           = 0
        self.ctl_dummy_fsm_op2_select_in = 0
        self.ctl_dummy_fsm_out_select_in = 0

        # input buses
        self.left_in_bus_thread0 = ["p"]*ROWS  # thread 0용
        self.left_in_bus_thread1 = ["p"]*ROWS  # thread 1용
        self.top_in_bus_thread0 = ["p"]*COLS   # thread 0용
        self.top_in_bus_thread1 = ["p"]*COLS   # thread 1용
        self.left_in_bus_thread2 = ["p"]*ROWS  # thread 2용
        self.top_in_bus_thread2 = ["p"]*COLS  # thread 2용
        self.left_in_bus_thread3 = ["p"]*ROWS  # thread 3용
        self.top_in_bus_thread3 = ["p"]*COLS  # thread 3용
        
        ##input -> queue -> margin element (when the margin pe of queue is full it is used)
        #not real queue 
        self.left_in_queue_thread0 = [Queue() for _ in range(ROWS)]  
        self.top_in_queue_thread0 = [Queue() for _ in range(COLS)]   
        self.left_in_queue_thread1 = [Queue() for _ in range(ROWS)]  
        self.top_in_queue_thread1 = [Queue() for _ in range(COLS)]   
        self.left_in_queue_thread2 = [Queue() for _ in range(ROWS)]  
        self.top_in_queue_thread2 = [Queue() for _ in range(COLS)]   
        self.left_in_queue_thread3 = [Queue() for _ in range(ROWS)]  
        self.top_in_queue_thread3 = [Queue() for _ in range(COLS)]  
        
        # output buses
        self.bottom_out_bus = [0]*COLS
        self.right_out_bus  = [0]*ROWS

        # 2D mac array with ArbitrationUnit
        self.mac_array = []
        for r in range(ROWS):
            row_macs = []
            for c in range(COLS):
                row_macs.append(ArbitrationUnit(word_size=WORD_SIZE,parent=self))
            self.mac_array.append(row_macs)
        

    def reset_all(self):
        for r in range(self.ROWS):
            self.left_in_bus_thread0[r] = 'p'
            self.left_in_bus_thread1[r] = 'p'
            self.left_in_bus_thread2[r] = 'p'
            self.left_in_bus_thread3[r] = 'p'
            self.right_out_bus[r] = 0
        for c in range(self.COLS):
            self.top_in_bus_thread0[c] = 'p'
            self.top_in_bus_thread1[c] = 'p'
            self.top_in_bus_thread2[c] = 'p'
            self.top_in_bus_thread3[c] = 'p'            
            self.bottom_out_bus[c] = 0
        for r in range(self.ROWS):
            for c in range(self.COLS):
                self.mac_array[r][c].reset()

    def set_control_signals(self, stat_bit_in, fsm_op2_sel, fsm_out_sel):
        self.ctl_stat_bit_in           = stat_bit_in
        self.ctl_dummy_fsm_op2_select_in = fsm_op2_sel
        self.ctl_dummy_fsm_out_select_in = fsm_out_sel

    def set_inputs(self, left_list1=None, top_list1=None, left_list2=None, top_list2=None, left_list3=None, top_list3=None, left_list4=None, top_list4=None):##checking##
        # Thread 0의 입력
        if left_list1 is not None and top_list1 is not None:
            for r in range(self.ROWS):
                self.left_in_bus_thread0[r] = left_list1[r]
            for c in range(self.COLS):
                self.top_in_bus_thread0[c] = top_list1[c]
            
        # Thread 1의 입력 (있는 경우)
        if left_list2 is not None and top_list2 is not None:
            for r in range(self.ROWS):
                self.left_in_bus_thread1[r] = left_list2[r]
            for c in range(self.COLS):
                self.top_in_bus_thread1[c] = top_list2[c]
        #print("set",left_list3,top_list3)
        # Thread 2의 입력 (있는 경우)
        if left_list3 is not None and top_list3 is not None:
            for r in range(self.ROWS):
                self.left_in_bus_thread2[r] = left_list3[r]
            for c in range(self.COLS):
                self.top_in_bus_thread2[c] = top_list3[c]
        # Thread 3의 입력 (있는 경우)
        if left_list4 is not None and top_list4 is not None:
            for r in range(self.ROWS):
                self.left_in_bus_thread3[r] = left_list4[r]
            for c in range(self.COLS):
                self.top_in_bus_thread3[c] = top_list4[c]
        self.top_in_bus_thread_list = [self.top_in_bus_thread0,self.top_in_bus_thread1,self.top_in_bus_thread2,self.top_in_bus_thread3]
        self.top_in_queue_thread_list = [self.top_in_queue_thread0,self.top_in_queue_thread1,self.top_in_queue_thread2,self.top_in_queue_thread3]
        self.left_in_bus_thread_list = [self.left_in_bus_thread0,self.left_in_bus_thread1,self.left_in_bus_thread2,self.left_in_bus_thread3]
        self.left_in_queue_thread_list = [self.left_in_queue_thread0,self.left_in_queue_thread1,self.left_in_queue_thread2,self.left_in_queue_thread3]
        

      
    def evaluate_one_cycle(self, cycle=0):
        temp_right_b = 0
        temp_bottom_b = 0
        # (1) 각 PE가 받을 lval/tval 계산 후 큐에 넣기
        '''시스톨릭 어레이가 값을 오른쪽 아래의 대각선 방향으로 전파하기 때문에
         시뮬레이션할 때는 반대방향으로 for loop을 하여 이전 pe에서 값들을 받아와서
         arbitration unit을 통해서 인접 pe로 데이터를 넘겨줌
         또한, 번호가 부여된 if문은 pe의 위치에 따라 구분함 
         3*3 size에서 (0,0)은 1 (0,1)은 2, (0,2)는 3
         (1,0)은 4 , (1,1)은 5 , (1,2)는 6, (2,0)는 7
         (2,1)은 8 (2,2)는 9로 정의함
         각 위치마다 입력데이터와 가중치 데이터를 받기도 하고, 인접 pe의 오른쪽과 아래쪽에 둘다 보내주어야 하는 경우와
         둘 중 하나만 보내줘도 되는 경우에 대해서 나누기 위함
         row나 col의 길이가 1인 경우에는 special case로 따로 정의함
         '''
        for t in range(self.ROWS + self.COLS-1,-1,-1):
            for r in range(self.ROWS):
                c = t - r
                if 0 <= c < self.COLS:
                    pe = self.mac_array[r][c]
                    if r == 0 and c == 0 and self.COLS>1 and self.ROWS>1: ## 1
                        for i in range(4):
                            if pe.top_queues[i].full():
                                pe.top_full_check[i]=1
                            if pe.left_queues[i].full():
                                pe.left_full_check[i]=1
                    
                            if self.top_in_bus_thread_list[i][c] != 'p':
                                self.top_in_queue_thread_list[i][c].put(self.top_in_bus_thread_list[i][c])
                                if not pe.top_queues[i].full():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                            else:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                    
                            if self.left_in_bus_thread_list[i][r] != 'p':    
                                self.left_in_queue_thread_list[i][r].put(self.left_in_bus_thread_list[i][r])
                                if not pe.left_queues[i].full():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                            else:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                        
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")
                        pe.arbitration_unit(r,c,"all")
                        
                        for i in range(4):##retry after arbitration
                            if pe.top_full_check[i] == 1:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                    
                            if pe.left_full_check[i] == 1:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                                                         
                        
                        pe.top_full_check[i] = [0,0,0,0]
                        pe.left_full_check[i] = [0,0,0,0]
                                    
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        if pe.chosen_thread != -1: 
                            self.mac_array[r+1][c].top_queues[pe.chosen_thread].put(pe.bottom_out)
                            self.mac_array[r][c+1].left_queues[pe.chosen_thread].put(pe.right_out)
                        
                        
                        
                        
                    elif r == 0 and 0 < c < self.COLS-1 and self.COLS>1 and self.ROWS>1: ## 2
                        for i in range(4):
                            if pe.top_queues[i].full():
                                pe.top_full_check[i]=1                            
                            if self.top_in_bus_thread_list[i][c] != 'p':
                                self.top_in_queue_thread_list[i][c].put(self.top_in_bus_thread_list[i][c])
                                if not pe.top_queues[i].full():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                            else:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                                
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")                        
                        
                        pe.arbitration_unit(r,c,"all")
                        if pe.chosen_thread != -1:
                            self.mac_array[r+1][c].top_queues[pe.chosen_thread].put(pe.bottom_out)
                            self.mac_array[r][c+1].left_queues[pe.chosen_thread].put(pe.right_out)
                    
                        for i in range(4):##retry after arbitration
                            if pe.top_full_check[i] == 1:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                        
                        pe.top_full_check[i] = [0,0,0,0]
                            
                            
                            
                                    
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")     
                    elif r == 0 and c == self.COLS-1 and self.COLS>1 and self.ROWS>1: ## 3
                        for i in range(4):
                            if pe.top_queues[i].full():
                                pe.top_full_check[i]=1                             
                            if self.top_in_bus_thread_list[i][c] != 'p':
                                self.top_in_queue_thread_list[i][c].put(self.top_in_bus_thread_list[i][c])
                                if not pe.top_queues[i].full():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                            else:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                                
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")
                        pe.arbitration_unit(r,c,"bottom")
                        
                        for i in range(4):##retry after arbitration
                            if pe.top_full_check[i] == 1:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())                        
                        pe.top_full_check[i] = [0,0,0,0]
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        if pe.chosen_thread != -1:
                            self.mac_array[r+1][c].top_queues[pe.chosen_thread].put(pe.bottom_out)




                    elif r == 0 and c == 0 and self.ROWS == 1 and self.COLS>1: #special case(row1) #1 
                        for i in range(4):
                            if pe.top_queues[i].full():
                                pe.top_full_check[i]=1
                            if pe.left_queues[i].full():
                                pe.left_full_check[i]=1
                    
                            if self.top_in_bus_thread_list[i][c] != 'p':
                                self.top_in_queue_thread_list[i][c].put(self.top_in_bus_thread_list[i][c])
                                if not pe.top_queues[i].full():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                            else:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                    
                            if self.left_in_bus_thread_list[i][r] != 'p':    
                                self.left_in_queue_thread_list[i][r].put(self.left_in_bus_thread_list[i][r])
                                if not pe.left_queues[i].full():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                            else:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                        

                        pe.arbitration_unit(r,c,"right")
                        
                        for i in range(4):##retry after arbitration
                            if pe.top_full_check[i] == 1:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                    
                            if pe.left_full_check[i] == 1:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                                                         
                        
                        pe.top_full_check[i] = [0,0,0,0]
                        pe.left_full_check[i] = [0,0,0,0]
                                    
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        if pe.chosen_thread != -1: 
                            self.mac_array[r][c+1].left_queues[pe.chosen_thread].put(pe.right_out)




                    elif r == 0 and 0 < c < self.COLS-1 and self.ROWS == 1 and self.COLS>1: ## special case(row1) #2
                        for i in range(4):
                            if pe.top_queues[i].full():
                                pe.top_full_check[i]=1                            
                            if self.top_in_bus_thread_list[i][c] != 'p':
                                self.top_in_queue_thread_list[i][c].put(self.top_in_bus_thread_list[i][c])
                                if not pe.top_queues[i].full():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                            else:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                                
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")                        
                        
                        pe.arbitration_unit(r,c,"right")
                        if pe.chosen_thread != -1:
                            self.mac_array[r][c+1].left_queues[pe.chosen_thread].put(pe.right_out)
                    
                        for i in range(4):##retry after arbitration
                            if pe.top_full_check[i] == 1:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                        
                        pe.top_full_check[i] = [0,0,0,0]
                    
                    
                    
                    
                    elif r == 0 and c == self.COLS-1 and self.ROWS == 1 and self.COLS>1: ## special case(row1) #3
                        for i in range(4):
                            if pe.top_queues[i].full():
                                pe.top_full_check[i]=1                            
                            if self.top_in_bus_thread_list[i][c] != 'p':
                                self.top_in_queue_thread_list[i][c].put(self.top_in_bus_thread_list[i][c])
                                if not pe.top_queues[i].full():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                            else:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                                
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")                        
                        
                        pe.arbitration_unit(r,c,"end")
                        
                        for i in range(4):##retry after arbitration
                            if pe.top_full_check[i] == 1:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                        
                        pe.top_full_check[i] = [0,0,0,0]
                    
                    
                    
                    
                    
                    elif r == 0 and c == 0 and self.COLS == 1 and self.ROWS >1: #special case(col1) #1 
                        for i in range(4):
                            if pe.top_queues[i].full():
                                pe.top_full_check[i]=1
                            if pe.left_queues[i].full():
                                pe.left_full_check[i]=1
                    
                            if self.top_in_bus_thread_list[i][c] != 'p':
                                self.top_in_queue_thread_list[i][c].put(self.top_in_bus_thread_list[i][c])
                                if not pe.top_queues[i].full():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                            else:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                    
                            if self.left_in_bus_thread_list[i][r] != 'p':    
                                self.left_in_queue_thread_list[i][r].put(self.left_in_bus_thread_list[i][r])
                                if not pe.left_queues[i].full():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                            else:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                        

                        pe.arbitration_unit(r,c,"bottom")
                        
                        for i in range(4):##retry after arbitration
                            if pe.top_full_check[i] == 1:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                    
                            if pe.left_full_check[i] == 1:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                                                         
                        
                        pe.top_full_check[i] = [0,0,0,0]
                        pe.left_full_check[i] = [0,0,0,0]
                                    
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        if pe.chosen_thread != -1: 
                            self.mac_array[r+1][c].top_queues[pe.chosen_thread].put(pe.bottom_out)
                    
                    
                    
                    
                    elif 0 < r < self.ROWS-1 and c == 0 and self.COLS ==1 and self.ROWS>1: ## special case(col1) #2
                        for i in range(4):
                            if pe.left_queues[i].full():
                                pe.left_full_check[i]=1                             
                            if self.left_in_bus_thread_list[i][r] != 'p':    
                                self.left_in_queue_thread_list[i][r].put(self.left_in_bus_thread_list[i][r])
                                if not pe.left_queues[i].full():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                            else:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                        
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")
                        pe.arbitration_unit(r,c,"bottom")
                        for i in range(4):##retry after arbitration
                            if pe.left_full_check[i] == 1:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                        pe.left_full_check[i] = [0,0,0,0]
                                                               
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        if pe.chosen_thread != -1:
                            self.mac_array[r+1][c].top_queues[pe.chosen_thread].put(pe.bottom_out)
                       
                       
                       
                                        
                    elif r == self.ROWS-1 and c == 0 and self.COLS ==1 and self.ROWS>1: ## special case(col1) #3
                        for i in range(4):
                            if pe.left_queues[i].full():
                                pe.left_full_check[i]=1                             
                            if self.left_in_bus_thread_list[i][r] != 'p':    
                                self.left_in_queue_thread_list[i][r].put(self.left_in_bus_thread_list[i][r])
                                if not pe.left_queues[i].full():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                            else:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                        
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")
                        pe.arbitration_unit(r,c,"end")
                        for i in range(4):##retry after arbitration
                            if pe.left_full_check[i] == 1:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                        pe.left_full_check[i] = [0,0,0,0]
                                                               
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")




                    elif r == 0 and c == 0 and self.ROWS == 1 and self.COLS ==1: #special case(row1 col1) #1 
                        for i in range(4):
                            if pe.top_queues[i].full():
                                pe.top_full_check[i]=1
                            if pe.left_queues[i].full():
                                pe.left_full_check[i]=1
                    
                            if self.top_in_bus_thread_list[i][c] != 'p':
                                self.top_in_queue_thread_list[i][c].put(self.top_in_bus_thread_list[i][c])
                                if not pe.top_queues[i].full():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                            else:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                    
                            if self.left_in_bus_thread_list[i][r] != 'p':    
                                self.left_in_queue_thread_list[i][r].put(self.left_in_bus_thread_list[i][r])
                                if not pe.left_queues[i].full():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                            else:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                        

                        pe.arbitration_unit(r,c,"end")
                        
                        for i in range(4):##retry after arbitration
                            if pe.top_full_check[i] == 1:
                                if not pe.top_queues[i].full() and not self.top_in_queue_thread_list[i][c].empty():
                                    pe.top_queues[i].put(self.top_in_queue_thread_list[i][c].get())
                                    
                            if pe.left_full_check[i] == 1:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                                                         
                        
                        pe.top_full_check[i] = [0,0,0,0]
                        pe.left_full_check[i] = [0,0,0,0]
                                    
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")


                    
                    
                    
                    elif 0 < r < self.ROWS-1 and c == self.COLS-1 and self.COLS>1 and self.ROWS>1: ##6
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")
                        pe.arbitration_unit(r,c,"bottom")
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        if pe.chosen_thread != -1:
                            self.mac_array[r+1][c].top_queues[pe.chosen_thread].put(pe.bottom_out)




                    elif 0 < r < self.ROWS-1 and c == 0 and self.COLS>1 and self.ROWS>1: ## 4
                        for i in range(4):
                            if pe.left_queues[i].full():
                                pe.left_full_check[i]=1                             
                            if self.left_in_bus_thread_list[i][r] != 'p':    
                                self.left_in_queue_thread_list[i][r].put(self.left_in_bus_thread_list[i][r])
                                if not pe.left_queues[i].full():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                            else:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                        
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")
                        pe.arbitration_unit(r,c,"all")
                        for i in range(4):##retry after arbitration
                            if pe.left_full_check[i] == 1:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                        pe.left_full_check[i] = [0,0,0,0]
                                                               
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        if pe.chosen_thread != -1:
                            self.mac_array[r+1][c].top_queues[pe.chosen_thread].put(pe.bottom_out)
                            self.mac_array[r][c+1].left_queues[pe.chosen_thread].put(pe.right_out)           
                    
                    
                    
                    
                    elif r == self.ROWS -1 and c == 0 and self.COLS>1 and self.ROWS>1: ## 7
                        for i in range(4):
                            if pe.left_queues[i].full():
                                pe.left_full_check[i]=1                             
                            if self.left_in_bus_thread_list[i][r] != 'p':    
                                self.left_in_queue_thread_list[i][r].put(self.left_in_bus_thread_list[i][r])
                                if not pe.left_queues[i].full():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())
                            else:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                        
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")
                        pe.arbitration_unit(r,c,"right")
                        for i in range(4):##retry after arbitration
                            if pe.left_full_check[i] == 1:
                                if not pe.left_queues[i].full() and not self.left_in_queue_thread_list[i][r].empty():
                                    pe.left_queues[i].put(self.left_in_queue_thread_list[i][r].get())                          
                        pe.left_full_check[i] = [0,0,0,0]                        
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        if pe.chosen_thread != -1:
                            self.mac_array[r][c+1].left_queues[pe.chosen_thread].put(pe.right_out)
             
             
             
             
                    elif r == self.ROWS -1 and 0 < c < self.COLS-1 and self.COLS>1 and self.ROWS>1: ##8
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")
                        pe.arbitration_unit(r,c,"right")
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        if pe.chosen_thread != -1:
                            self.mac_array[r][c+1].left_queues[pe.chosen_thread].put(pe.right_out)
                        
                        
                        
                        
                        
                    elif r == self.ROWS -1 and c == self.COLS-1 and self.COLS>1 and self.ROWS>1: ##9
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")
                        pe.arbitration_unit(r,c,"end")
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        
                        
                        
                        
                    else: ## 5
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"before arbitration")
                        pe.arbitration_unit(r,c,"all")
                        #status_print(r,c,pe.left_queues,pe.top_queues,pe.chosen_thread,pe.bypassed_threads,"after arbitration")
                        if pe.chosen_thread != -1:
                            self.mac_array[r+1][c].top_queues[pe.chosen_thread].put(pe.bottom_out)
                            self.mac_array[r][c+1].left_queues[pe.chosen_thread].put(pe.right_out)
                    
                    
                    
                    ##bypass processing
                    for b in range(len(self.mac_array[r][c].bypassed_threads)):## if bypass thread list empty, it pass
                        bypass_thread = self.mac_array[r][c].bypassed_threads[b]
                        temp_right_b = self.mac_array[r][c].right_out_bypass[b]
                        temp_bottom_b = self.mac_array[r][c].bottom_out_bypass[b]
                        if r != self.ROWS-1 and c == self.COLS-1 and self.COLS>1 and self.ROWS>1:
                            pe = self.mac_array[r+1][c]
                            pe.top_queues[bypass_thread].put(temp_bottom_b)
                        elif r == self.ROWS-1 and c != self.COLS-1 and self.COLS>1 and self.ROWS>1:
                            pe = self.mac_array[r][c+1]
                            pe.left_queues[bypass_thread].put(temp_right_b)
                        elif r != self.ROWS-1 and c != self.COLS-1 and self.COLS>1 and self.ROWS>1:
                            pe = self.mac_array[r+1][c]
                            pe.top_queues[bypass_thread].put(temp_bottom_b)
                            pe = self.mac_array[r][c+1]
                            pe.left_queues[bypass_thread].put(temp_right_b)
                        elif 0<=c<self.COLS-1 and self.ROWS==1 and self.COLS>1:
                            pe = self.mac_array[r][c+1]
                            pe.left_queues[bypass_thread].put(temp_right_b)
                        elif 0<=r<self.ROWS-1 and self.COLS==1 and self.ROWS>1:
                            pe = self.mac_array[r+1][c]
                            pe.top_queues[bypass_thread].put(temp_bottom_b)
                        else:
                            pass

        # 마지막 행/열의 출력만 bus에 기록
        for c in range(self.COLS):
            self.bottom_out_bus[c] = self.mac_array[self.ROWS-1][c].bottom_out

        # right_out_bus[r] = PE(r, COLS-1).right_out
        for r in range(self.ROWS):
            self.right_out_bus[r] = self.mac_array[r][self.COLS-1].right_out
    
    def get_outputs(self):
        return (self.bottom_out_bus, self.right_out_bus)


###############################
# 3) 상위 제어 로직 (Scheduler/Controller)
#    "평행사변형(parallelogram)" 데이터 주입
###############################
def parallelogram_controller(systolic, A, B, num_cycles):
    M, K = A.shape
    K2, N = B.shape
    assert K==K2, "A,B 곱 가능해야"
    
    # A, B 행렬을 스레드별로 분할
    quarter_k = K // 4
    A1 = A[:, :quarter_k]
    A2 = A[:, quarter_k:2*quarter_k]
    A3 = A[:, 2*quarter_k:3*quarter_k]
    A4 = A[:, 3*quarter_k:]
    B1 = B[:quarter_k, :]
    B2 = B[quarter_k:2*quarter_k, :]
    B3 = B[2*quarter_k:3*quarter_k, :]
    B4 = B[3*quarter_k:, :]
    systolic.reset_all()

    for cycle in range(num_cycles):
        # Thread 1 입력값 계산 (A1, B1 사용)
        new_left1 = ["p"]*systolic.ROWS
        new_top1 = ["p"]*systolic.COLS
        
        # Thread 2 입력값 계산 (A2, B2 사용)
        new_left2 = ["p"]*systolic.ROWS
        new_top2 = ["p"]*systolic.COLS
        
        # Thread 3 입력값 계산 (A3, B3 사용)
        new_left3 = ["p"]*systolic.ROWS
        new_top3 = ["p"]*systolic.COLS
        
        # Thread 4 입력값 계산 (A4, B4 사용)
        new_left4 = ["p"]*systolic.ROWS
        new_top4 = ["p"]*systolic.COLS
        
        # 평행사변형 패턴으로 Thread 1 데이터 주입
        for r in range(systolic.ROWS):
            if 0 <= cycle - r < A1.shape[1]:  # A1의 열 수만큼
                new_left1[r] = A1[r, cycle-r].item()
        for c in range(systolic.COLS):
            if 0 <= cycle - c < B1.shape[0]:  # B1의 행 수만큼
                new_top1[c] = B1[cycle-c, c].item()

        # 평행사변형 패턴으로 Thread 2 데이터 주입
        for r in range(systolic.ROWS):
            if 0 <= cycle - r < A2.shape[1]:  # A2의 열 수만큼
                new_left2[r] = A2[r, cycle-r].item()
        
        for c in range(systolic.COLS):
            if 0 <= cycle - c < B2.shape[0]:  # B2의 행 수만큼
                new_top2[c] = B2[cycle-c, c].item()

        # 평행사변형 패턴으로 Thread 3 데이터 주입
        for r in range(systolic.ROWS):
            if 0 <= cycle - r < A3.shape[1]:  # A2의 열 수만큼
                new_left3[r] = A3[r, cycle-r].item()
        
        for c in range(systolic.COLS):
            if 0 <= cycle - c < B3.shape[0]:  # B2의 행 수만큼
                new_top3[c] = B3[cycle-c, c].item()

        # 평행사변형 패턴으로 Thread 4 데이터 주입
        for r in range(systolic.ROWS):
            if 0 <= cycle - r < A4.shape[1]:  # A2의 열 수만큼
                new_left4[r] = A4[r, cycle-r].item()
        
        for c in range(systolic.COLS):
            if 0 <= cycle - c < B4.shape[0]:  # B2의 행 수만큼
                new_top4[c] = B4[cycle-c, c].item()

        systolic.set_control_signals(
            stat_bit_in=0,
            fsm_op2_sel=1,
            fsm_out_sel=0  # 항상 B값 전달 모드
        )

        systolic.set_inputs(new_left1, new_top1, new_left2, new_top2, 
                            new_left3, new_top3, new_left4, new_top4)
        # 여기에 cycle 인자를 추가:
        systolic.evaluate_one_cycle(cycle=cycle)

        # 각 PE의 현재 상태 출력
        checking_not_empty = 0
        for r in range(systolic.ROWS):
            row_accum = []
            for c in range(systolic.COLS):
                pe = systolic.mac_array[r][c]
                row_accum.append(pe.accumulator_reg)
                #모든 pe의 queue가 비었다면 cycle 측정을 마침
                if (len(pe.left_queues[0].queue)==0 and len(pe.left_queues[1].queue)==0 and pe.bottom_out==0 and\
                    len(pe.top_queues[0].queue)==0 and len(pe.top_queues[1].queue)==0 and pe.right_out==0 and\
                    len(pe.left_queues[2].queue)==0 and len(pe.left_queues[3].queue)==0 and \
                    len(pe.top_queues[2].queue)==0 and len(pe.top_queues[3].queue)==0 and \
                    len(pe.right_out_bypass)==0 and len(pe.bottom_out_bypass)==0 and \
                    all(q.empty() for q in systolic.left_in_queue_thread0) and all(q.empty() for q in systolic.left_in_queue_thread1) and \
                    all(q.empty() for q in systolic.left_in_queue_thread2) and all(q.empty() for q in systolic.left_in_queue_thread3) and \
                    all(q.empty() for q in systolic.top_in_queue_thread0) and all(q.empty() for q in systolic.top_in_queue_thread1) and \
                    all(q.empty() for q in systolic.top_in_queue_thread2) and all(q.empty() for q in systolic.top_in_queue_thread3)
                    ):
                    checking_not_empty = checking_not_empty
                else:
                    checking_not_empty += 1
        
        if checking_not_empty == 0:
            return cycle+1



#----------------------------------------------------------------------#

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
    


    def execute(self, a_addr: int, b_addr: int, seq_len: int,rowA:int, colA: int, colB:int) -> int:

        self.num_real = 0
        a_shape = (rowA, seq_len)
        b_shape = (seq_len, colB)

        a_size = self.arr_height * seq_len * self.itemsize
        b_size = seq_len * self.arr_width * self.itemsize
        if not self.check_onchip_mem_access(a_addr, a_size):
            raise Exception("[ERROR] onchip memory request for A is invalid.")
        if not self.check_onchip_mem_access(b_addr, b_size):
            raise Exception("[ERROR] onchip memory request for B is invalid.")

    # 1) On-chip 메모리에서 A, B를 읽어옴
        A = self.read_onchip_mem(a_addr, a_shape)
        B = self.read_onchip_mem(b_addr, b_shape)

    # 2) 임시 SystolicArray (6x6)
        systolic = Systolicarray(ROWS=rowA, COLS=colB, WORD_SIZE=16)
        systolic.reset_all()
        systolic.set_control_signals(stat_bit_in=1, fsm_op2_sel=1, fsm_out_sel=1)

    # 3) 필요한 사이클 계산 (M, K, N은 이 함수 밖에서 정의한 값을 쓰면 안 됨)
    #    여기서는 seq_len, arr_height, arr_width 기반으로 가정하거나,
    #    혹은 A.shape, B.shape를 직접 써서 계산해야 합니다.
    #    만약 M,K,N을 계속 쓰고 싶으면, execute() 인자로 넘겨주거나
    #    전역(global) 변수로 둬야 함.
    #    예: (단순 시뮬) num_cycles = (a_shape[0] + b_shape[1] + a_shape[1]) - 1 + 10
    #    또는 여기서는 예시로만 임의로 해두겠습니다.
        M_ = A.size(0)      # A의 행
        K_ = A.size(1)      # A의 열(= B의 행)
        N_ = B.size(1)      # B의 열
        systolic = Systolicarray(ROWS=rowA, COLS=colB, WORD_SIZE=16)
        systolic.reset_all()
        num_cycles = rowA + colA + colB - 1 + 10

    # 4) Systolic 연산
        num_real = parallelogram_controller(systolic, A, B, num_cycles)
        #print(num_real)

    # 5) Systolic 결과(각 PE.accumulator_reg)를
    #    현재 SystolicArrayOS의 acc_registers에 반영(누적)
        for r in range(rowA):
            for c in range(colB):
                self.acc_registers[r, c] += systolic.mac_array[r][c].accumulator_reg

    # 아주 간단히, "걸린 시간"을 seq_len + arr_height + arr_width - 2 라고 가정
        return num_real

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

def simulate_systolic_array(a: torch.Tensor, b: torch.Tensor, csv_file: str = "alexnet_T4Q2.csv"):
    # 아래 코드는 전체 매트릭스 곱을 타일 단위로 시뮬레이션하는 예시
    sa = SystolicArrayOS(arr_height=16, arr_width=16, dtype=torch.float32)
    M, K = a.shape
    K2, N = b.shape
    
    csv_file = "alexnet_T4Q2.csv"
    headers = ["M", "K", "N", "OScycle", "cycle", "Speedup", "Validation", "note"]
    note = "fc2/pin"

    print(f"\nMatrix sizes:")
    print(f"A: {M}x{K}")
    print(f"B: {K}x{N}")
    print(f"Systolic Array size: {M}x{N}")

    output_matrix = torch.zeros((M, N), dtype=torch.float32)

    tile_height = 16
    tile_width  = 16
    timestamp   = 0
    compute_cycle = 0
    original_OS_cycle = 0
    # We'll define on-chip addresses for A, B, D
    a_addr = 0
    b_addr = 8 * 8 * 4000   # 256 bytes offset (8x8 float)
    d_addr = 8 * 8 * 8000   # 512

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
            original_OS_cycle += rowA + colB + K - 2
            compute_cycle += sa.execute(a_addr, b_addr, seq_len,rowA, colA, colB)
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
    reference = torch.matmul(a, b)
    validation = "passed" if torch.allclose(reference, output_matrix) else "failed"
    print(f"simulated computation time: {compute_cycle}")
    print(f"simulated execution time: {timestamp}")
    print(f"result validation: {'passed' if torch.allclose(reference, output_matrix) else 'failed'}")
    print("reference:")
    print(reference)
    print("simulated:")
    print(output_matrix)
    print("Speedup:")
    print(speedup)

    # CSV에 결과 저장
    save_to_csv(csv_file, headers, [M, K, N, original_OS_cycle, compute_cycle, speedup, validation, note])
    return output_matrix

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

if __name__ == "__main__":
    # STEP 1: create CNN model
    #   - You can download the CNN model from torchvision
    #   - Model downloaded from Huggingface is also compatible with this example code
    
    import torchvision
    
    weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1.IMAGENET1K_V1
    model = torchvision.models.alexnet(weights=weights)
    model.eval()

    pruning_ratios = {
        "features.0": 0.16,   # Conv1
        "features.3": 0.62,   # Conv2
        "features.6": 0.65,   # Conv3
        "features.8": 0.63,   # Conv4
        "features.10": 0.63,  # Conv5
        "classifier.1": 0.91, # FC1
        "classifier.4": 0.91, # FC2
        "classifier.6": 0.75, # FC3
    }

    # Apply pruning
    for name, module in model.named_modules():
        if name in pruning_ratios:
            prune.l1_unstructured(module, name='weight', amount=pruning_ratios[name])
            prune.remove(module, 'weight')  # Remove pruning mask to apply actual pruning

    
    # STEP 2: create extraction session and register hook to the model
    #   - Use 'lowering' option to convert the Conv2d operator into GEMM (General Matrix Multiplication)
    #   - It is recommended to activate the 'lowering' option to simulate with the systolic array behavioral model
    
    session = ExtractionHookSession(verbose=True)
    session.register_hook(module=model, lowering=True)
    
    
    # STEP 3: inference with the data
    #   - Instead of using the zeros tensor, use the imagenet data
    
    
    #x = torch.zeros((1, 3, 224, 224))    # instead of using the zeros tensor, use the imagenet data

    imagenet_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # AlexNet 입력 크기로 조정
        transforms.ToTensor(),  # Tensor로 변환
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화
        
    ])

    imagenet_path = "/data/datasets/ILSVRC2012"

    val_path = os.path.join(imagenet_path, "val")
    pin_class = "n04285008"

    #sample_class = os.listdir(val_path)[0]  # 첫 번째 클래스를 선택
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

    #  추출된 ifm과 wgt에 대해서 systolic array 진행
    # case : 특정 layer 선택해서 진행
    conv2d_layers = []
    Linear_layers = []
    
    for layer_name in session.data.keys():
        if ".ifm" in layer_name:
            # 레이어 타입을 확인
            original_layer_name = layer_name.replace("AlexNet.", "").split(".ifm")[0]
            print(original_layer_name)
            layer_type = session.get_layer_type_by_name(model, original_layer_name)
            if layer_type == torch.nn.Conv2d:
                conv2d_layers.append(layer_name)
            elif layer_type == torch.nn.Linear:
                Linear_layers.append(layer_name)

    print(f"Conv2d layers: {conv2d_layers}\nLinear layers: {Linear_layers}")

    # 특정 레이어 선택 (예: Conv2d 1번 레이어)
    selected_layer_type = "Linear"  # "conv2d" 또는 "Linear" 선택
    selected_layer_index = 2  # 선택한 레이어의 번호 (1부터 시작)

    if selected_layer_type == "conv2d":
        selected_layer = conv2d_layers[selected_layer_index - 1]
    elif selected_layer_type == "Linear":
        selected_layer = Linear_layers[selected_layer_index - 1]
    else:
        raise ValueError("Invalid layer type selected. Use 'conv2d' or 'Linear'.")

    # 선택된 레이어에 대해 Systolic Array 수행
    ifm = session.data[selected_layer]
    wgt = session.data[selected_layer.replace(".ifm", ".wgt")]
    if "classifier" in selected_layer:  # Fully Connected (Linear) 계층이면 Transpose 적용
        wgt = wgt.t()
    # Systolic Array 적용
    systolic_output = simulate_systolic_array(ifm, wgt)

    # 결과 출력
    print(f"Layer: {selected_layer.replace('.ifm', '')}")
    print(f"Systolic Array Output Shape: {systolic_output.shape}")


from collections import OrderedDict
import numpy as np
from graphviz import Digraph
from sklearn.mixture import GaussianMixture
import time
from collections import defaultdict
class AbstractModel():
    def __init__(self):
        self.initial = []
        self.final = []
        

        
class DTMC(AbstractModel):
    '''
    Multiple DTMCs from a set of sets of traces
    traces: a set of sets of traces
    '''
    def __init__(self, min_val, max_val, grid_num, clipped=True):
        super().__init__()
        self.min = min_val
        self.max = max_val
        self.k = grid_num
        self.dim = max_val.shape[0]
        self.total_states = pow(grid_num,self.dim)
        self.unit = (max_val - min_val) / self.k
        self.clipped = clipped

    def state_abstract(self, con_states):
        lower_bound = self.min
        upper_bound = self.max
        unit = (upper_bound - lower_bound)/self.k
        abs_states = np.zeros(con_states.shape[0],dtype=np.int)
        tmp = ((con_states-self.min)/unit).astype(int)
        if self.clipped:
            tmp = np.clip(tmp, 0, self.k-1)
            
        dims = tmp.shape[1]
        for i in range(dims):
            abs_states = abs_states + tmp[:,i]*pow(self.k, i)
#         abs_states = np.expand_dims(abs_states,axis=-1)
            
        abs_states = [str(item) for item in abs_states]
        return abs_states
    
    def extract_abs_trace(self,rewards,abs_states,num=None):
        end_idx = np.where(np.abs(rewards)==1)[0]
        traces = []
        results = []
        tracesLen = []
        success_count = 0
        fail_count = 0
        if num is not None:
            end_idx = end_idx[:num]  
            print(">>>Extract first {} games".format(num))
        count = 0
        for cur_end in end_idx:
            cur_trace = []
            cur_count = 0
            while True:
                cur_state = str(abs_states[count])
                cur_trace.append(cur_state)
                if cur_count == 0 and cur_state not in self.initial:
                    self.initial.append(cur_state)
                count += 1
                cur_count += 1
                if count == cur_end:
                    if cur_state not in self.final:
                        self.final.append(cur_state)
                    if rewards[cur_end] == 1:
                        cur_trace.append('S')
                        success_count += 1
                        results.append(1)
                    elif rewards[cur_end] == -1:
                        cur_trace.append('F')
                        fail_count += 1
                        results.append(0)
                    else:
                        assert False,"count:{},reward:{}".format(count, rewards[cur_end])
                    tracesLen.append(cur_count)
                    traces.append(cur_trace)
                    break
                    
        print("Total traces:{}".format(len(traces)))
        print("Success:{}".format(success_count))
        print("Failure:{}".format(fail_count))
        return traces,np.array(tracesLen),np.array(results)

    
    
    
    def profiling_all_traces(self, all_traces):
        state_dic = defaultdict(list) # {state:[fail_stat, success_stat]}
        edge_dic = defaultdict(list)
        for cur_trace in all_traces:
            isSuccess =  cur_trace[-1]=='S'
            for idx in range(len(cur_trace)-1):
                cur_state = cur_trace[idx]
                next_state = cur_trace[idx+1]
                cur_edge = (cur_state,next_state)
                if cur_state not in state_dic:
                    state_dic[cur_state] = [0,0]
                state_dic[cur_state][isSuccess] += 1
                if cur_edge not in edge_dic:
                    edge_dic[cur_edge] = [0,0]
                edge_dic[cur_edge][isSuccess] += 1
        return state_dic,edge_dic
    

    
    
    
   
    


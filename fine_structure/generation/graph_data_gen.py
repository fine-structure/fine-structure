import numpy as np
import numpy 
import random

def get_state(state_vector):
    for i, e in enumerate(state_vector):
        if e != 0:
            return i

def get_next_state(super_position_state, weight_precision = 1000):
     weight_sum = sum(super_position_state)
     weighted_state = (e * weight_precision / weight_sum for e in super_position_state)

     state_list = []
     i = 0
     for item in weighted_state:
         state_list += [ i ] * int(item)
         i += 1

     new_state = [0.] * len(super_position_state)
     new_state[random.choice(state_list)] = 1.0;

     return new_state

def walk(N, init_state, transition_m):
     state = init_state
     for i in range(N):
         super_position_state = numpy.dot(state, transition_m)
         state = get_next_state(super_position_state)
         yield state
            
            
          
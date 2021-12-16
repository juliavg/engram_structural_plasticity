import numpy as np
import random as rd
import h5py as h5

NE = 10000
stimulated_fraction = 0.1
stimulated_pop = int(NE*stimulated_fraction)

stimuli_file = h5.File("stimuli.hdf5",'w')

unconditional_stimulus = np.array(np.random.choice(np.arange(NE),int(stimulated_pop),replace=False)).astype(int)
conditional_stimulus1 = np.array(np.random.choice(np.arange(NE)[np.logical_not(np.in1d(np.arange(NE),unconditional_stimulus))],int(stimulated_pop),replace=False)).astype(int)
conditional_stimulus2 = np.array(np.random.choice(np.arange(NE)[np.logical_not(np.in1d(np.arange(NE),np.concatenate((unconditional_stimulus,conditional_stimulus1))))],int(stimulated_pop),replace=False)).astype(int)

stimuli_file.create_dataset('unconditional_stimulus',unconditional_stimulus.shape,dtype=unconditional_stimulus.dtype)
stimuli_file.create_dataset('conditional_stimulus1',conditional_stimulus1.shape,dtype=conditional_stimulus1.dtype)
stimuli_file.create_dataset('conditional_stimulus2',conditional_stimulus2.shape,dtype=conditional_stimulus2.dtype)
stimuli_file['unconditional_stimulus'][...] = unconditional_stimulus
stimuli_file['conditional_stimulus1'][...] = conditional_stimulus1
stimuli_file['conditional_stimulus2'][...] = conditional_stimulus2

stimuli_file.close()

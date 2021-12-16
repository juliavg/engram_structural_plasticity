import numpy as np
import sys
import nest
import h5py as h5
import matplotlib.pyplot as plt

direc = sys.argv[0].rpartition('/')[0]+"/"

# Define patterns
stimuli_file = h5.File(direc+"stimuli.hdf5",'r')
unconditional_stimulus = np.array(stimuli_file['unconditional_stimulus'])
conditional_stimulus1 = np.array(stimuli_file['conditional_stimulus1'])
conditional_stimulus2 = np.array(stimuli_file['conditional_stimulus2'])
stimuli_file.close()

us_cs1 = np.concatenate((unconditional_stimulus,conditional_stimulus1))

direc = direc+"data/"

###################################################################################
# Parameters
###################################################################################

# General simulation parameters
n_vp = 100  # total number of virtual processes
dt = 0.1   # simulation resolution (ms)
MSP_update_interval = 10   # update interval for MSP (ms)

# Parameters for asynchronous irregular firing
g = 8.0                # ratio between maximum amplitude of EPSP and EPSP
eta = 1.5                 # ratio between external rate and external frequency needed for the mean input to reach threshold in absence of feedback
eps = 0.1                 # connection probability for static connections (all but EE)
order = 2500                 # order of network size
NE = 4*order             # number of excitatory neurons
NI = 1*order             # number of inhibitory neurons
N = NE+NI               # total number of neurons
CE = int(eps*NE)         # number of incoming excitatory synapses per inhibitory neuron
CI = int(eps*NI)         # number of incominb inhibitory synapses per neuron  

# Growth
growth_time = 100000.            # growth time (ms)
growth_step = 1000.
cicles = int(growth_time/growth_step)                # cicles for recording during growth

# Training
stimuli_order = np.array([1., 1., 0., 0., 1., 0.])
# Stimulation
stimulation_time = 2000.   # stimulation time (ms)
stimulation_step = 1000.
stimulation_cicles = int(stimulation_time/stimulation_step)       # cicles for recording during stimulation
stimulation_strength = 1.4      # modulation of external input firing rate during stimulation
# Post stimulation
post_stimulation_time = 48000. # time post stimulation (ms)
post_stimulation_step = 1000.
post_stimulation_cicles = int(post_stimulation_time/post_stimulation_step)   # cicles for recording post stimulation

# Decay
decay_time = 100000.
decay_step = 1000.
decay_cicles = int(decay_time/decay_step)


# Parameters of the integrate and fire neuron
neuron_model = "iaf_psc_delta"
CMem = 250.0                # membrane capacitance (pF)
tauMem = 20.0                 # membrane time constant (ms)
theta = 20.0                 # spike threshold (mV)
t_ref = 2.                   # refractory period (ms)
E_L = 0.                   # resting membrane potential (mV)
V_reset = 10.                  # reset potential of the membrane (mV)
V_m = 0.                   # initial membrane potential (mV)
tau_Ca = 1000.               # time constant for calcium trace (ms)
beta_Ca = 1./tau_Ca            # increment on calcium trace per spike (1/ms)
J = 0.1                  # postsynaptic amplitude in mV
delay = 1.                   # synaptic delay (ms)

neuron_params   = {
                    "C_m"       : CMem,
                    "tau_m"     : tauMem,
                    "t_ref"     : t_ref,
                    "E_L"       : E_L,
                    "V_reset"   : V_reset,
                    "V_m"       : V_m,
                    "beta_Ca"   : beta_Ca,
                    "tau_Ca"    : tau_Ca,
                    "V_th"      : theta
                   }

# External input rate
nu_th = theta/(J*CE*tauMem)
nu_ex = eta*nu_th
rate = 1000.0*nu_ex*CE
 
# Parameter for structural plasticity
growth_curve = "linear"            # type of growth curve for synaptic elements
z0 = 1.                  # initial number of synaptic elements
slope = -2.5                # slope of growth curve for synaptic elements
synapse_model = "static_synapse"    # plastic EE synapse type

target_rate = 0.008


# Simulation setup
###################################################################################

def simulate_cicle(simulate_steps):
    step = np.diff(simulate_steps)[0]
    for simulation_time in simulate_steps:
        nest.Simulate(step)

        events = nest.GetStatus(spike_detector,'events')[0]
        times = events['times']
        senders = events['senders']
        nest.SetStatus(spike_detector,'n_events',0)

        local_connections = nest.GetConnections()
        sources = np.array(nest.GetStatus(local_connections,'source'))
        targets = np.array(nest.GetStatus(local_connections,'target'))

        extension = str(simulation_time)+"_"+str(rank)+".npy"
        np.save(direc+"times_"+extension,times)
        np.save(direc+"senders_"+extension,senders)
        np.save(direc+"sources_"+extension,sources)
        np.save(direc+"targets_"+extension,targets)

seed = 2*59+341
seeds = np.arange(1,n_vp+1,1)+seed

# Set Kernel
nest.ResetKernel()

rank = nest.Rank()

nest.EnableStructuralPlasticity()
nest.SetKernelStatus({"total_num_virtual_procs" : n_vp,
                      "resolution" : dt, 
                      "print_time" : False,
                      "structural_plasticity_update_interval" : int(MSP_update_interval/dt),   # update interval for MSP in time steps
                      "grng_seed" : seed,
                      "rng_seeds" : list(seeds)
                      })

np.save(direc+"grng_seed_"+str(seed)+".npy",seed)
np.save(direc+"rng_seeds_"+str(seed)+".npy",seeds)

# Set model defaults
nest.SetDefaults(neuron_model, neuron_params)
nest.CopyModel(neuron_model, 'excitatory')
nest.CopyModel(neuron_model, 'inhibitory')
nest.CopyModel("static_synapse","device",{"weight":J, "delay":delay})
nest.CopyModel("static_synapse","inhibitory_synapse",{"weight":-g*J, "delay":delay})
nest.CopyModel("static_synapse","EI_synapse",{"weight":J, "delay":delay})
nest.CopyModel(synapse_model, 'msp_excitatory')
nest.SetDefaults('msp_excitatory',{'weight': J,'delay': delay})


# Assign synaptic elements with growth curve to excitatory neuron model
gc_den  = {'growth_curve': growth_curve, 'z': z0, 'growth_rate': -slope*target_rate, 'eps': target_rate, 'continuous': False}
gc_axon = {'growth_curve': growth_curve, 'z': z0, 'growth_rate': -slope*target_rate, 'eps': target_rate, 'continuous': False}
nest.SetDefaults('excitatory', 'synaptic_elements', {'Axon_exc': gc_axon, 'Den_exc': gc_den})

# Use SetKernelStatus to activate the plastic synapses
nest.SetKernelStatus({
    'structural_plasticity_synapses': {
        'syn1': {
            'model': 'msp_excitatory',
            'post_synaptic_element': 'Den_exc',
            'pre_synaptic_element': 'Axon_exc',
        }
    },
    'autapses': False,
})

# Create nodes
pop_exc = nest.Create('excitatory', NE)
pop_inh = nest.Create('inhibitory', NI)
poisson_generator_ex = nest.Create('poisson_generator',NE)
poisson_generator_inh = nest.Create('poisson_generator')
spike_detector = nest.Create("spike_detector")

nest.SetStatus(poisson_generator_ex, {"rate": rate})
nest.SetStatus(poisson_generator_inh, {"rate": rate})
nest.SetStatus(spike_detector,{"withtime": True, "withgid": True})

# Connect nodes
nest.Connect(pop_exc, pop_inh,{'rule': 'fixed_indegree','indegree': CE},'EI_synapse')
nest.Connect(pop_inh, pop_exc+pop_inh,{'rule': 'fixed_indegree','indegree': CI},'inhibitory_synapse')
nest.Connect(poisson_generator_ex, pop_exc,'one_to_one', model="device")
nest.Connect(poisson_generator_inh, pop_inh,'all_to_all',model="device")
nest.Connect(pop_exc+pop_inh, spike_detector,'all_to_all',model="device")


###################################################################################
# Simulate
###################################################################################

# Grow network
growth_steps = np.arange(growth_step,growth_time+1.,growth_step)
simulate_cicle(growth_steps)

# Baseline
# US
nest.SetStatus(list(np.array(poisson_generator_ex)[unconditional_stimulus.astype(int)]), {"rate": rate*stimulation_strength})
stimulation_end = growth_time + stimulation_time
stimulation_steps = np.arange(growth_time+stimulation_step, stimulation_end+1., stimulation_step)
simulate_cicle(stimulation_steps)

post_stimulation_end = stimulation_end + post_stimulation_time
post_stimulation_steps = np.arange(stimulation_end+post_stimulation_step, post_stimulation_end+1., post_stimulation_step)
nest.SetStatus(poisson_generator_ex, {"rate": rate})
simulate_cicle(post_stimulation_steps)

# CS1
nest.SetStatus(list(np.array(poisson_generator_ex)[conditional_stimulus1.astype(int)]), {"rate": rate*stimulation_strength})
stimulation_end = post_stimulation_end + stimulation_time
stimulation_steps = np.arange(post_stimulation_end+stimulation_step, stimulation_end+1., stimulation_step)
simulate_cicle(stimulation_steps)

post_stimulation_end = stimulation_end + post_stimulation_time
post_stimulation_steps = np.arange(stimulation_end+post_stimulation_step, post_stimulation_end+1., post_stimulation_step)
nest.SetStatus(poisson_generator_ex, {"rate": rate})
simulate_cicle(post_stimulation_steps)

# CS2
nest.SetStatus(list(np.array(poisson_generator_ex)[conditional_stimulus2.astype(int)]), {"rate": rate*stimulation_strength})
stimulation_end = post_stimulation_end + stimulation_time
stimulation_steps = np.arange(post_stimulation_end+stimulation_step, stimulation_end+1., stimulation_step)
simulate_cicle(stimulation_steps)

baseline_end = stimulation_end + post_stimulation_time
post_stimulation_steps = np.arange(stimulation_end+post_stimulation_step, baseline_end+1., post_stimulation_step)
nest.SetStatus(poisson_generator_ex, {"rate": rate})
simulate_cicle(post_stimulation_steps)


# Stimulate - training
for ii,stimulus in enumerate(stimuli_order):
    stimulation_start = baseline_end+ii*(stimulation_time+post_stimulation_time)
    stimulation_end = baseline_end + (ii+1)*stimulation_time + ii*post_stimulation_time
    stimulation_steps = np.arange(stimulation_start+stimulation_step, stimulation_end+1., stimulation_step)

    if stimulus == 0:
        pattern = us_cs1*1
    else:
        pattern = conditional_stimulus2*1

    nest.SetStatus(list(np.array(poisson_generator_ex)[pattern.astype(int)]), {"rate": rate*stimulation_strength})
    simulate_cicle(stimulation_steps)

    post_stimulation_end = stimulation_end + post_stimulation_time
    post_stimulation_steps = np.arange(stimulation_end+post_stimulation_step, post_stimulation_end+1., post_stimulation_step)
    nest.SetStatus(poisson_generator_ex, {"rate": rate})
    simulate_cicle(post_stimulation_steps)

# Decay
decay_start = baseline_end+stimuli_order.shape[0]*(stimulation_time+post_stimulation_time)
decay_end = decay_start+decay_time
growth_steps = np.arange(decay_start+decay_step,decay_end+1.,decay_step)
simulate_cicle(growth_steps)

# Reconsolidation 1
stimulation_end = decay_end+stimulation_time
growth_steps = np.arange(decay_end+stimulation_step,stimulation_end+1.,stimulation_step)
nest.SetStatus(list(np.array(poisson_generator_ex)[conditional_stimulus1.astype(int)]), {"rate": rate*stimulation_strength})
simulate_cicle(growth_steps)

post_stimulation_end = stimulation_end+post_stimulation_time
growth_steps = np.arange(stimulation_end+post_stimulation_step, post_stimulation_end+1., post_stimulation_step)
nest.SetStatus(poisson_generator_ex, {"rate": rate})
simulate_cicle(growth_steps)

# Reconsolidation 2
stimulation_end = post_stimulation_end+stimulation_time
growth_steps = np.arange(post_stimulation_end+stimulation_step,stimulation_end+1.,stimulation_step)
nest.SetStatus(list(np.array(poisson_generator_ex)[conditional_stimulus2.astype(int)]), {"rate": rate*stimulation_strength})
simulate_cicle(growth_steps)

post_stimulation_end = stimulation_end+post_stimulation_time
growth_steps = np.arange(stimulation_end+post_stimulation_step, post_stimulation_end+1., post_stimulation_step)
nest.SetStatus(poisson_generator_ex, {"rate": rate})
simulate_cicle(growth_steps)

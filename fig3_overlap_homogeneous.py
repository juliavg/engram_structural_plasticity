import numpy as np
import sys
import nest
import random as rd
import scipy.sparse as sp

direc = sys.argv[0].rpartition('/')[0]+'/'
overlap_idx = int(sys.argv[1])


NE = 10000
NI = 2500
N  = NE+NI
nu_ext = 15000.

stimulated_pop           = 1000
stimulus_strength        = 1.05
stimulation_time         = 10000.
between_stimulation_time = 5000.
binsize_similarity       = 10.
overlap_values           = np.arange(101)*0.01
n_trials                 = 50
time_after_n_trials      = n_trials*(between_stimulation_time+stimulation_time)
overlap = overlap_values[overlap_idx]

bins       = np.arange(0,stimulation_time+binsize_similarity,binsize_similarity)

connection_probability = 0.1

# General parameters of the integrate and fire neuron
neuron_model = "iaf_psc_delta"
CMem         = 250.0                # membrane capacitance (pF)
t_ref        = 2.                   # refractory period (ms)
E_L          = 0.                   # resting membrane potential (mV)
V_reset      = 10.                  # reset potential of the membrane (mV)
V_m          = 0.                   # initial membrane potential (mV)
V_th         = 20.
delay        = 1.                   # synaptic delay (ms)

# Paramters population
tau_mem_pop = 20.                 # membrane time constant (ms)
J           = 0.1                  # postsynaptic amplitude in mV
g           = 8

population_params   = {
                "C_m"       : CMem,
                "tau_m"     : tau_mem_pop,
                "t_ref"     : t_ref,
                "E_L"       : E_L,
                "V_reset"   : V_reset,
                "V_m"       : V_m,
                "V_th"      : V_th
               }

neurons_pattern     = np.arange(stimulated_pop)
comparison_pattern  = np.zeros(NE)
comparison_pattern[neurons_pattern] = 1

def produce_spk_count(N,times,senders):
    spk_count = np.zeros((N,int(stimulation_time/binsize_similarity)))
    for ss in np.arange(N):
        times_ss = times[senders==ss+1]
        spk_count[ss,:] = np.histogram(times_ss,bins=bins)[0]
    spk_count[spk_count!=0] = 1
    return spk_count

for ii in np.arange(n_trials):
    nest.ResetKernel()

    nest.SetKernelStatus({'grng_seed':np.random.randint(0,1000),'rng_seeds':(np.random.randint(0,1000),)})

    nest.CopyModel("static_synapse","device",{"weight":J,"delay":delay})

    exc_pop         = nest.Create(neuron_model,NE)
    inh_pop         = nest.Create(neuron_model,NI)
    poisson_generator = nest.Create("poisson_generator",NE+1)
    spike_detector  = nest.Create("spike_detector")

    nest.SetStatus(exc_pop,population_params)
    nest.SetStatus(inh_pop,population_params)
    nest.SetStatus(poisson_generator,{'rate':nu_ext})

    nest.Connect(exc_pop,exc_pop+inh_pop,{'rule':'fixed_indegree','indegree':int(connection_probability*NE)},{'weight':J,'delay':delay})
    nest.Connect(inh_pop,exc_pop+inh_pop,{'rule':'fixed_indegree','indegree':int(connection_probability*NI)},{'weight':-g*J,'delay':delay})
    nest.Connect(poisson_generator[:NE],exc_pop,'one_to_one',model='device')
    nest.Connect([poisson_generator[-1]],inh_pop,'all_to_all',model='device')
    nest.Connect(exc_pop,spike_detector)

    # Simulate partial patterns
    nest.Simulate(between_stimulation_time)
    nest.SetStatus(spike_detector,'n_events',0)

    neurons_stimulated = rd.sample(np.arange(stimulated_pop),int(overlap*stimulated_pop))
    nest.SetStatus(list(np.array(poisson_generator)[neurons_stimulated]),{'rate':stimulus_strength*nu_ext})
    nest.Simulate(stimulation_time)
    events  = nest.GetStatus(spike_detector,'events')[0]
    senders = events['senders']
    times   = events['times']
    nest.SetStatus(spike_detector,'n_events',0)
    nest.SetStatus(poisson_generator,{'rate':nu_ext})

    spk_count   = produce_spk_count(NE,times-between_stimulation_time,senders)
    similarity  = np.sum(spk_count*(comparison_pattern-np.mean(comparison_pattern))[np.newaxis].T,axis=0) / (np.mean(comparison_pattern)*NE*(1-np.mean(comparison_pattern)))

    np.save("similarity_overlap"+str(overlap_idx)+"_trial"+str(ii)+".npy",similarity)

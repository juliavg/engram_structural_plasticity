import numpy as np
import time
import sys

par     = __import__(sys.argv[1].rpartition("/")[-1].partition(".")[0])
seed    = int(sys.argv[2])
direc   = sys.argv[1].rpartition('/')[0]+"/data/"

import nest 

rank = nest.Rank()

grng_seed               = seed
rng_seeds               = range(seed+1,seed+par.total_num_virtual_procs+1)

np.save(direc+"global_seeds.npy",grng_seed)
np.save(direc+"thread_seeds.npy",rng_seeds)

nest.ResetKernel()

nest.EnableStructuralPlasticity()

nest.SetKernelStatus({"resolution": par.dt, "print_time": False})

nest.SetKernelStatus({
    'structural_plasticity_update_interval' : int(par.MSP_update_interval/par.dt),               # update interval for MSP in time steps
    'total_num_virtual_procs'       : par.total_num_virtual_procs,
    'grng_seed'                     : grng_seed,
    'rng_seeds'                     : rng_seeds,
})

nest.SetDefaults(par.neuron_model, par.neuron_params)

# Create generic neuron with Axon and Dendrite
nest.CopyModel(par.neuron_model, 'excitatory')
nest.CopyModel(par.neuron_model, 'inhibitory')

# growth curves
gc_den = {'growth_curve': par.growth_curve_d, 'z': par.z0_mean, 'growth_rate': par.slope_d*par.eps, 'eps': par.eps,
          'continuous': False,'tau_vacant':par.tau_vacant}
gc_axon = {'growth_curve': par.growth_curve_a, 'z': par.z0_mean, 'growth_rate': par.slope_a*par.eps, 'eps': par.eps,
           'continuous': False,'tau_vacant':par.tau_vacant}

nest.SetDefaults('excitatory', 'synaptic_elements', {'Axon_exc': gc_axon, 'Den_exc': gc_den})

# Create synapse models
nest.CopyModel('static_synapse','device',{'weight':par.weight, 'delay':par.delay})
nest.CopyModel('static_synapse','inhibitory_synapse',{'weight':-par.g*par.weight, 'delay':par.delay})
nest.CopyModel("static_synapse","EI_synapse",{"weight":par.weight, "delay":par.delay})
nest.CopyModel(par.synapse_model, 'msp_excitatory',{'delay': par.delay,'weight': par.weight})

# Use SetKernelStatus to activate the synapse model
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


# build network
pop_exc = nest.Create('excitatory', par.NE)
pop_inh = nest.Create('inhibitory', par.NI)

poisson_generator_inh = nest.Create('poisson_generator')
nest.SetStatus(poisson_generator_inh, {"rate": par.rate})
nest.Connect(poisson_generator_inh, pop_inh,'all_to_all',model='device')

poisson_generator_ex = nest.Create('poisson_generator',2)
nest.SetStatus(poisson_generator_ex,{'rate': par.rate})
nest.Connect([poisson_generator_ex[0]], pop_exc[:par.subgroup_size],'all_to_all', model='device')
nest.Connect([poisson_generator_ex[1]], pop_exc[par.subgroup_size:],'all_to_all', model='device')

spike_detector = nest.Create("spike_detector")
nest.SetStatus(spike_detector,{
                                "withtime"  : True,
                                "withgid"   : True,
                                "stop"      : par.growth_time+par.stimulation_time+par.post_stimulation_time
                                })
nest.Connect(pop_exc+pop_inh, spike_detector,'all_to_all',model="device")

nest.Connect(pop_exc,pop_inh,{'rule': 'fixed_indegree','indegree': par.CE},'EI_synapse')
nest.Connect(pop_inh,pop_exc+pop_inh,{'rule': 'fixed_indegree','indegree': par.CI},'inhibitory_synapse')

def simulate_cicle(growth_steps):
    growth_step = growth_steps[1]-growth_steps[0]
    for simulation_time in growth_steps:
        nest.Simulate(growth_step)

        local_connections = nest.GetConnections(pop_exc+pop_inh, pop_exc+pop_inh)
        sources = nest.GetStatus(local_connections,'source')
        targets = nest.GetStatus(local_connections,'target')

        events = nest.GetStatus(spike_detector,'events')[0]
        times = events['times']
        senders = events['senders']

        extension = str(simulation_time+growth_step)+"_"+str(rank)+".npy"
        np.save(direc+"times_"+extension,times)
        np.save(direc+"senders_"+extension,senders)
        nest.SetStatus(spike_detector,'n_events',0)

        del local_connections

        np.save(direc+"sources_"+extension,sources)
        np.save(direc+"targets_"+extension,targets)

        loc_e = [stat['global_id'] for stat in nest.GetStatus(pop_exc)  if stat['local']]
        np.save(direc+"loc_e_"+extension,loc_e)

        calcium = np.array(nest.GetStatus(loc_e,'Ca'))
        np.save(direc+"calcium_"+extension,calcium)

        synaptic_elements = nest.GetStatus(loc_e,'synaptic_elements')
        z_axon = np.zeros(len(synaptic_elements))
        z_dend = np.zeros(len(synaptic_elements))
        for ee,elements in enumerate(synaptic_elements):
            z_axon[ee] = elements['Axon_exc']['z']
            z_dend[ee] = elements['Den_exc']['z']
        np.save(direc+"z_axon_"+extension,z_axon)
        np.save(direc+"z_dend_"+extension,z_dend)

# Grow network
growth_steps = np.arange(par.cicles)*par.pre_step
simulate_cicle(growth_steps)

# Stimulate network
nest.SetStatus([poisson_generator_ex[0]],{"rate": (1+par.mu)*par.rate})
growth_steps = par.growth_time+np.arange(par.stimulation_cicles)*par.stimulation_step
simulate_cicle(growth_steps)

# Post stimulation
nest.SetStatus([poisson_generator_ex[0]],{"rate": par.rate})
growth_steps = par.growth_time+par.stimulation_time+np.arange(par.cicles)*par.post_stimulation_step
simulate_cicle(growth_steps)

growth_steps = par.growth_time+par.stimulation_time+par.post_stimulation_time+np.arange(par.cicles)*par.post_stimulation_step2
for simulation_time in growth_steps:
    nest.Simulate(par.post_stimulation_step2)

    local_connections = nest.GetConnections(pop_exc+pop_inh, pop_exc+pop_inh)
    sources = nest.GetStatus(local_connections,'source')
    targets = nest.GetStatus(local_connections,'target')

    extension = str(simulation_time+par.post_stimulation_step2)+"_"+str(rank)+".npy"
    
    del local_connections

    np.save(direc+"sources_"+extension,sources)
    np.save(direc+"targets_"+extension,targets)

    loc_e = [stat['global_id'] for stat in nest.GetStatus(pop_exc)  if stat['local']]
    np.save(direc+"loc_e_"+extension,loc_e)

    calcium = np.array(nest.GetStatus(loc_e,'Ca'))
    np.save(direc+"calcium_"+extension,calcium)

    synaptic_elements = nest.GetStatus(loc_e,'synaptic_elements')
    z_axon = np.zeros(len(synaptic_elements))
    z_dend = np.zeros(len(synaptic_elements))
    for ee,elements in enumerate(synaptic_elements):
        z_axon[ee] = elements['Axon_exc']['z']
        z_dend[ee] = elements['Den_exc']['z']
    np.save(direc+"z_axon_"+extension,z_axon)
    np.save(direc+"z_dend_"+extension,z_dend)

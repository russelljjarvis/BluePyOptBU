
# coding: utf-8

# # Creating a simple cell optimisation
#
# This notebook will explain how to set up an optimisation of simple single compartmental cell with two free parameters that need to be optimised

# In[ ]:

# import matplotlib.pyplot as iplt
# plt.rcParams['lines.linewidth'] = 2
# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload')
import os


# First we need to import the module that contains all the functionality to create electrical cell models

# In[ ]:

import bluepyopt as bpop
import bluepyopt.ephys as ephys


# If you want to see a lot of information about the internals,
# the verbose level can be set to 'debug' by commenting out
# the following lines

# In[ ]:

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Setting up a cell template
# -------------------------
# First a template that will describe the cell has to be defined. A template consists of:
# * a morphology
# * model mechanisms
# * model parameters
#
# ### Creating a morphology
# A morphology can be loaded from a file (SWC or ASC).

# In[ ]:

morph = ephys.morphologies.NrnFileMorphology('simple.swc')


# By default a Neuron morphology has the following sectionlists: somatic, axonal, apical and basal. Let's create an object that points to the somatic sectionlist. This object will be used later to specify where mechanisms have to be added etc.

# In[ ]:

somatic_loc = ephys.locations.NrnSeclistLocation('somatic', seclist_name='somatic')


# ### Creating a mechanism
#
# Now we can add ion channels to this morphology. Let's add the default Neuron Hodgkin-Huxley mechanism to the soma.

# In[ ]:

hh_mech = ephys.mechanisms.NrnMODMechanism(
        name='hh',
        prefix='hh',
        locations=[somatic_loc])


# The 'name' field can be chosen by the user, this name should be unique. The 'prefix' points to the same field in the NMODL file of the channel. 'locations' specifies which sections the mechanism will be added to.
#
# ### Creating parameters
#
# Next we need to specify the parameters of the model. A parameter can be in two states: frozen and not-frozen. When a parameter is frozen it has an exact value, otherwise it only has some bounds but the exact value is not known yet.
# Let's first a parameter that sets the capitance of the soma to a frozen value

# In[ ]:

cm_param = ephys.parameters.NrnSectionParameter(
        name='cm',
        param_name='cm',
        value=1.0,
        locations=[somatic_loc],
        frozen=True)


# And parameters that represent the maximal conductance of the sodium and potassium channels. These two parameters will be optimised later.

# In[ ]:

gnabar_param = ephys.parameters.NrnSectionParameter(
        name='gnabar_hh',
        param_name='gnabar_hh',
        locations=[somatic_loc],
        bounds=[0, .2],
        frozen=False)
gkbar_param = ephys.parameters.NrnSectionParameter(
        name='gkbar_hh',
        param_name='gkbar_hh',
        bounds=[0, .1],
        locations=[somatic_loc],
        frozen=False)


# ### Creating the template
#
# To create the cell template, we pass all these objects to the constructor of the template

# In[ ]:

simple_cell = ephys.models.CellModel(
        name='simple_cell',
        morph=morph,
        mechs=[hh_mech],
        params=[cm_param, gnabar_param, gkbar_param])


# Now we can print out a description of the cell

# In[ ]:

print simple_cell


# With this cell we can build a cell evaluator.

# ## Setting up a cell evaluator
#
# To optimise the parameters of the cell we need to create cell evaluator object.
# This object will need to know which protocols to injection, which parameters to optimise, etc.

# ### Creating the protocols
#
# A protocol consists of a set of stimuli, and a set of responses (i.e. recordings). These responses will later be used by a calculate
# the score of the parameter values.
# Let's create two protocols, two square current pulse at somatic[0](0.5) with different amplitudes.
# We first need to create a location object

# In[ ]:

soma_loc = ephys.locations.NrnSeclistCompLocation(
        name='soma',
        seclist_name='somatic',
        sec_index=0,
        comp_x=0.5)


# and then the stimuli, recordings and protocols. For each protocol we add a recording and a stimulus in the soma.

# In[ ]:

sweep_protocols = []
for protocol_name, amplitude in [('step1', 0.01), ('step2', 0.05)]:
    stim = ephys.stimuli.NrnSquarePulse(
                step_amplitude=amplitude,
                step_delay=100,
                step_duration=50,
                location=soma_loc,
                total_duration=200)
    rec = ephys.recordings.CompRecording(
            name='%s.soma.v' % protocol_name,
            location=soma_loc,
            variable='v')
    protocol = ephys.protocols.SweepProtocol(protocol_name, [stim], [rec])
    sweep_protocols.append(protocol)
twostep_protocol = ephys.protocols.SequenceProtocol('twostep', protocols=sweep_protocols)


# ### Running protocols on a cell
#
# Now we're at a stage where we can actually run a protocol on the cell.

# In[ ]:

nrn = ephys.simulators.NrnSimulator()


# The run() method of a protocol accepts a cell model, a set of parameter values and a simulator

# In[ ]:

default_params = {'gnabar_hh': 0.1, 'gkbar_hh': 0.03}
responses = twostep_protocol.run(cell_model=simple_cell, param_values=default_params, sim=nrn)


# Plotting the response traces is now easy:

# In[ ]:

'''
def plot_responses(responses):
    plt.subplot(2,1,1)
    plt.plot(responses['step1.soma.v']['time'], responses['step1.soma.v']['voltage'], label='step1')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(responses['step2.soma.v']['time'], responses['step2.soma.v']['voltage'], label='step2')
    plt.legend()
    plt.tight_layout()

plot_responses(responses)
plt.show()
'''

# As you can see, when we use different parameter values, the response looks different.

# In[ ]:

other_params = {'gnabar_hh': 0.11, 'gkbar_hh': 0.04}
twostep_protocol.run(cell_model=simple_cell, param_values=other_params, sim=nrn)


# In[ ]:




# ### Defining eFeatures and objectives
#
# For every response we need to define a set of eFeatures we will use for the fitness calculation later. We have to combine features together into objectives that will be used by the optimalisation algorithm. In this case we will create one objective per feature:

# In[ ]:

efel_feature_means = {'step1': {'Spikecount': 1}, 'step2': {'Spikecount': 5}}

objectives = []

for protocol in sweep_protocols:
    stim_start = protocol.stimuli[0].step_delay
    stim_end = stim_start + protocol.stimuli[0].step_duration
    for efel_feature_name, mean in efel_feature_means[protocol.name].iteritems():
        feature_name = '%s.%s' % (protocol.name, efel_feature_name)
        feature = ephys.efeatures.eFELFeature(
                    feature_name,
                    efel_feature_name=efel_feature_name,
                    recording_names={'': '%s.soma.v' % protocol.name},
                    stim_start=stim_start,
                    stim_end=stim_end,
                    exp_mean=mean,
                    exp_std=0.05 * mean)
        objective = ephys.objectives.SingletonObjective(
            feature_name,
            feature)
        objectives.append(objective)


# ### Creating the cell evaluator
#
# We will need an object that can use these objective definitions to calculate the scores from a protocol response. This is called a ScoreCalculator.

# In[ ]:

score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)


# Combining everything together we have a CellEvaluator. The CellEvaluator constructor has a field 'parameter_names' which contains the (ordered) list of names of the parameters that are used as input (and will be fitted later on).

# In[ ]:

cell_evaluator = ephys.evaluators.CellEvaluator(
        cell_model=simple_cell,
        param_names=['gnabar_hh', 'gkbar_hh'],
        fitness_protocols={twostep_protocol.name: twostep_protocol},
        fitness_calculator=score_calc,
        sim=nrn,
        isolate_protocols=False)


# ### Evaluating the cell
#
# The cell can now be evaluate for a certain set of parameter values.

# In[ ]:

print cell_evaluator.evaluate_with_dicts(default_params)
print cell_evaluator.evaluate_with_dicts(other_params)


# ## Setting up and running an optimisation
#
# Now that we have a cell template and an evaluator for this cell, we can set up an optimisation.

# And this optimisation can be run for a certain number of generations

# In[ ]:

optimisation = bpop.optimisations.DEAPOptimisation(
    evaluator=cell_evaluator,
    offspring_size = 100,
    seed=1)

# results = optimisation.run(max_ngen=10, cp_filename='checkpoints/checkpoint.pkl')


# The optimisation has return us 4 objects: final population, hall of fame, statistical logs and history.
#
# The final population contains a list of tuples, with each tuple representing the two parameters of the model

# In[ ]:

import pickle
# pickle.dump(results, open('results.pkl', 'w'))
# results = pickle.load(open('results.pkl')

cp = pickle.load(open('checkpoints/checkpoint.pkl'))
results = (cp['population'],
        cp['halloffame'],
        cp['logbook'],
        cp['history'])


pop, hall_of_fame, logs, hist = results


# The best individual found during the optimisation is the first individual of the hall of fame

# In[ ]:

all_inds = hist.genealogy_history.values()
best_inds = [ind for ind in all_inds if ind.fitness.valid and ind.fitness.sum == 0]
print [ind.fitness.values for ind in best_inds]
# best_inds = hall_of_fame[:100]
best_ind = best_inds[1]
print 'Best individual: ', best_ind
print 'Fitness values: ', best_ind.fitness.values


# We can evaluate this individual and make use of a convenience function of the cell evaluator to return us a dict of the parameters

# In[ ]:

for best_ind in best_inds[:100]:
    best_ind_dict = cell_evaluator.param_dict(best_ind)
    print cell_evaluator.evaluate_with_dicts(best_ind_dict), best_ind
    twostep_protocol.run(cell_model=simple_cell, param_values=best_ind_dict, sim=nrn)


# As you can see the evaluation returns the same values as the fitness values provided by the optimisation output.
# We can have a look at the responses now.

# In[ ]:

import pickle
RESPONSE_PICKLE = 'responses.pkl'

if os.path.exists(RESPONSE_PICKLE):
    responses = pickle.load(open(RESPONSE_PICKLE))
else:
    responses = []
    for best_ind in best_inds:
       best_ind_dict = cell_evaluator.param_dict(best_ind)
       responses.append(cell_evaluator.run_protocols(cell_evaluator.fitness_protocols.values(), param_values=best_ind_dict))
    pickle.dump(responses, open(RESPONSE_PICKLE, 'w'))



# Let's have a look at the optimisation statistics.
# We can plot the minimal score (sum of all objective scores) found in every optimisation.
# The optimisation algorithm uses negative fitness scores, so we actually have to look at the maximum values log.

# In[ ]:
'''
import numpy

def get_slice(start, end, data):
    return slice(numpy.searchsorted(data, start),
                 numpy.searchsorted(data, end))

gen_numbers = logs.select('gen')
min_fitness = numpy.array(logs.select('min'))
max_fitness = logs.select('max')
mean_fitness = numpy.array(logs.select('avg'))
std_fitness = numpy.array(logs.select('std'))

fig, ax = plt.subplots(3, figsize=(10, 10), facecolor='white')
fig_trip, ax_trip = plt.subplots(1, figsize=(10, 5), facecolor='white')

plot_count = len(responses)
for index, response in enumerate(responses[:plot_count]):
    color='lightblue'
    if index == plot_count - 1:
        color='blue'

    # best_ind_dict = cell_evaluator.param_dict(best_ind)
    # responses = cell_evaluator.run_protocols(cell_evaluator.fitness_protocols.values(), param_values=best_ind_dict)
    sl = get_slice(80, 200, response['step1.soma.v']['time'])
    ax[0].plot(response['step1.soma.v']['time'][sl], response['step1.soma.v']['voltage'][sl], color=color, linewidth=1)
    sl = get_slice(80, 200, response['step2.soma.v']['time'])
    axes = ax[1].plot(response['step2.soma.v']['time'][sl], response['step2.soma.v']['voltage'][sl], color=color, linewidth=1)
    # axes[0].set_rasterized(True)

ax[0].set_xlabel('Time (ms)')
ax[0].set_ylabel('Voltage (mV)')
ax[1].set_xlabel('Time (ms)')
ax[1].set_ylabel('Voltage (mV)')

# ax[0].legend()

std = std_fitness
mean = mean_fitness
minimum = min_fitness
stdminus = mean - std
stdplus = mean + std

ax[2].plot(
    gen_numbers,
    mean,
    color='black',
    linewidth=2,
    label='population average')

ax[2].fill_between(
    gen_numbers,
    stdminus,
    stdplus,
    color='lightgray',
    linewidth=2,
    label=r'population standard deviation')

ax[2].plot(
    gen_numbers,
    minimum,
    color='red',
    linewidth=2,
    label='population minimum')

ax[2].set_xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
ax[2].set_xlabel('Generation #')
ax[2].set_ylabel('Sum of objectives')
ax[2].set_ylim([0, max(stdplus)])
ax[2].legend()

all_inds = hist.genealogy_history.values()

gnabars = numpy.array([ind[0] for ind in all_inds])
gkbars = numpy.array([ind[1] for ind in all_inds])
sums = numpy.array([ind.fitness.sum for ind in all_inds])
logsums = numpy.log(sums+1.0)
# psums = zip(gnabars, gkbars, sums)
zero_gnabars = gnabars[numpy.where(sums == 0)]
zero_gkbars = gkbars[numpy.where(sums == 0)]

# X = numpy.linspace(gnabar_param.bounds[0], gnabar_param.bounds[1], 150)
# Y = numpy.linspace(gkbar_param.bounds[0], gkbar_param.bounds[1], 150)
# X,Y = numpy.meshgrid(X,Y)
# import matplotlib
# import scipy.interpolate
# Z1 = scipy.interpolate.griddata(numpy.vstack((gnabars.flatten(), gkbars.flatten())).T, numpy.vstack(sums.flatten()), (X, Y), method='linear').reshape(X.shape)
# Z = matplotlib.mlab.griddata(gnabars, gkbars, sums, X, Y, interp='linear')
# Z1m = numpy.ma.masked_where(numpy.isnan(Z1),Z1)
# mesh_axes = ax[2].pcolormesh(X,Y,Z1m)

# plt.plot(gnabars, gkbars, '.', color='k')

trip_axis = ax_trip.tripcolor(gnabars,gkbars,logsums,20)
# plt.tricontourf(gnabars,gkbars,sums,20)
# cbar_ax = fig.add_axes([0.85, 0.29, 0.025, 0.2])
fig_trip.colorbar(trip_axis, label='log(sum of objectives + 1)')
ax_trip.set_xlabel('Parameter gnabar')
ax_trip.set_ylabel('Parameter gkbar')

plot_axis = ax_trip.plot(list(zero_gnabars), list(zero_gkbars), 'o', color='lightblue')
# ax[4].set_xlabel('Parameter gnabar')
# ax[4].set_ylabel('Parameter gkbar')
# ax[4].set_xlim(gnabar_param.bounds)
# ax[4].set_ylim(gkbar_param.bounds)
# fig.colorbar(trip_axis, ax=ax[3])

fig.tight_layout()
fig_trip.tight_layout()

# fig_trip.subplots_adjust(right=0.8, hspace=0.4)

# print X, Y
# print Z

fig.savefig('figures/simplecell_traces.eps')
fig_trip.savefig('figures/simplecell_trip.eps')

fig.show()
fig_trip.show()


# In[ ]:
'''



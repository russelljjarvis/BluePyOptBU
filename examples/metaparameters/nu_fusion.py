#!/usr/bin/env python
# coding: utf-8

# # Creating an optimisation with meta parameters
# 
# This notebook will explain how to set up an optimisation that uses metaparameters (parameters that control other parameters)


import bluepyopt as bpop
import bluepyopt.ephys as ephys
import pickle
from sciunit.scores import ZScore
from sciunit import TestSuite
from sciunit.scores.collections import ScoreArray
import sciunit
import numpy as np
from neuronunit.optimisation.optimization_management import dtc_to_rheo, switch_logic,active_values
from neuronunit.tests.base import AMPL, DELAY, DURATION

import quantities as pq
PASSIVE_DURATION = 500.0*pq.ms
PASSIVE_DELAY = 200.0*pq.ms


import matplotlib.pyplot as plt

from bluepyopt.ephys.models import ReducedCellModel




from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from neuronunit.optimisation.optimization_management import test_all_objective_test
from neuronunit.optimisation.optimization_management import check_binary_match, three_step_protocol,inject_and_plot_passive_model


simple_cell = ephys.models.ReducedCellModel(
        name='simple_cell',
        params=MODEL_PARAMS["IZHI"],backend="IZHI")  
simple_cell.backend = "IZHI"


# First we need to import the module that contains all the functionality to create electrical cell models

# If you want to see a lot of information about the internals, 
# the verbose level can be set to 'debug' by commenting out
# the following lines

# Setting up the cell
# ---------------------
# This is very similar to the simplecell example in the directory above. For a more detailed explanation, look there.

# For this example we will create two separate parameters to store the specific capacitance. One for the soma and one for the soma. We will put a metaparameter on top of these two to keep the value between soma and axon the same.

# The metaparameter, the one that will be optimised, will make sure the two parameters above keep always the same value

# And parameters that represent the maximal conductance of the sodium and potassium channels. These two parameters will be not be optimised but are frozen.

# ### Creating the template
# 
# To create the cell template, we pass all these objects to the constructor of the template.
# We *only* put the metaparameter, not its subparameters.

# In[6]:



from neuronunit.optimisation.model_parameters import MODEL_PARAMS
import numpy as np

simple_cell = ephys.models.ReducedCellModel(
        name='simple_cell',
        params=MODEL_PARAMS["IZHI"],backend="IZHI")  
simple_cell.backend = "IZHI"


# Now we can print out a description of the cell

# In[7]:


model = simple_cell
model.params = {k:np.mean(v) for k,v in model.params.items() }


# With this cell we can build a cell evaluator.

# ## Setting up a cell evaluator
# 
# To optimise the parameters of the cell we need to create cell evaluator object. 
# This object will need to know which protocols to inject, which parameters to optimise, etc.

# ### Creating the protocols
# 
# 

# In[ ]:





# In[8]:


#twostep_protocol


# ### Running a protocol on a cell
# 
# Now we're at a stage where we can actually run a protocol on the cell. We first need to create a Simulator object.

# In[9]:




# The run() method of a protocol accepts a cell model, a set of parameter values and a simulator

# Plotting the response traces is now easy:

# ### Defining eFeatures and objectives
# 
# For every response we need to define a set of eFeatures we will use for the fitness calculation later. We have to combine features together into objectives that will be used by the optimalisation algorithm. In this case we will create one objective per feature:

# In[13]:



tests = pickle.load(open("processed_multicellular_constraints.p","rb"))
nu_tests = tests['Hippocampus CA1 pyramidal cell'].tests
nu_tests[0].score_type = ZScore

nu_tests, OM, target = test_all_objective_test(MODEL_PARAMS["IZHI"],model_type="IZHI",protocol={'allen':False,'elephant':True})


# In[14]:


nu_tests = list(nu_tests.values())
nu_tests[0].score_type = ZScore
target = three_step_protocol(target)
target.rheobase
target.everything


# In[ ]:


get_ipython().run_cell_magic('capture', '', "'''\nefel_feature_means = {'step1': {'Spikecount': 1}, 'step2': {'Spikecount': 5}}\n\nobjectives = []\n\nfor protocol in sweep_protocols:\n    stim_start = protocol.stimuli[0].step_delay\n    stim_end = stim_start + protocol.stimuli[0].step_duration\n    for efel_feature_name, mean in efel_feature_means[protocol.name].items():\n        feature_name = '%s.%s' % (protocol.name, efel_feature_name)\n        feature = ephys.efeatures.eFELFeature(\n                    feature_name,\n                    efel_feature_name=efel_feature_name,\n                    recording_names={'': '%s.soma.v' % protocol.name},\n                    stim_start=stim_start,\n                    stim_end=stim_end,\n                    exp_mean=mean,\n                    exp_std=0.05 * mean)\n        objective = ephys.objectives.SingletonObjective(\n            feature_name,\n            feature)\n        objectives.append(objective)\n        \nscore_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)    \n'''")


# In[ ]:


get_ipython().run_cell_magic('capture', '', "\n\nsweep_protocols = []\nfor protocol_name, amplitude in [('step1', 0.05)]:\n    vm = model.inject_and_plot_model(\n        DELAY=100,\n        DURATION=500    \n    )\n    amplitude = model.rheobase\n    stim = ephys.stimuli.NrnSquarePulse(\n            step_amplitude=amplitude,\n            step_delay=100,\n            step_duration=500,\n            total_duration=1000)\n\n    rec = model.vm\n    \n    protocol = ephys.protocols.SweepProtocol(protocol_name, [stim], [rec])\n    sweep_protocols.append(protocol)\ntwostep_protocol = ephys.protocols.SequenceProtocol('twostep', protocols=sweep_protocols)")


# In[ ]:



def initialise_test(v,rheobase):
    v = switch_logic([v])
    v = v[0]
    k = v.name
    if not hasattr(v,'params'):
        v.params = {}
    if not 'injected_square_current' in v.params.keys():    
        v.params['injected_square_current'] = {}
    if v.passive == False and v.active == True:
        keyed = v.params['injected_square_current']
        v.params = active_values(keyed,rheobase)
        v.params['injected_square_current']['delay'] = DELAY
        v.params['injected_square_current']['duration'] = DURATION
    if v.passive == True and v.active == False:

        v.params['injected_square_current']['amplitude'] =  -10*pq.pA
        v.params['injected_square_current']['delay'] = PASSIVE_DELAY
        v.params['injected_square_current']['duration'] = PASSIVE_DURATION

    if v.name in str('RestingPotentialTest'):
        v.params['injected_square_current']['delay'] = PASSIVE_DELAY
        v.params['injected_square_current']['duration'] = PASSIVE_DURATION
        v.params['injected_square_current']['amplitude'] = 0.0*pq.pA    
        
    return v

class NUFeature(object):
    def __init__(self,test,model):
        self.test = test
        self.model = model
    def calculate_score(self,responses):
        model = responses['model'].dtc_to_model()
        model.attrs = responses['params']
        self.test = initialise_test(self.test,responses['rheobase'])
        if "Rheobase" in str(self.test.name):
            prediction = {'value':responses['rheobase']}

            score_gene = self.test.compute_score(self.test.observation,prediction)

        try:
            score_gene = self.test.judge(model)
        except:
            return 100.0

        if not isinstance(type(score_gene),type(None)):
            if not isinstance(score_gene,sciunit.scores.InsufficientDataScore):
                if not isinstance(type(score_gene.log_norm_score),type(None)):
                    try:

                        lns = np.abs(score_gene.log_norm_score)
                    except:
                        # works 1/2 time that log_norm_score does not work
                        # more informative than nominal bad score 100
                        lns = np.abs(score_gene.raw)
                else:
                    # works 1/2 time that log_norm_score does not work
                    # more informative than nominal bad score 100

                    lns = np.abs(score_gene.raw)    
            else:
                lns = 100
        if lns==np.inf:
            lns = 100
        #print(lns,self.test.name)
        return lns

    
objectives = []
#for protocol in sweep_protocols:
#    stim_start = protocol.stimuli[0].step_delay
#    stim_end = stim_start + protocol.stimuli[0].step_duration
    
for tt in nu_tests:
    feature_name = '%s.%s' % (tt.name, tt.name)
    ft = NUFeature(tt,model)
    objective = ephys.objectives.SingletonObjective(
        feature_name,
        ft)
    objectives.append(objective)

score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives) 
        
        
#objectives[0]        



# ### Creating the cell evaluator
# 
# We will need an object that can use these objective definitions to calculate the scores from a protocol response. This is called a ScoreCalculator.

# In[ ]:


get_ipython().run_cell_magic('capture', '', "'''\nfrom sciunit import TestSuite\nfrom sciunit.scores.collections import ScoreArray\nimport sciunit\nimport numpy as np\nfrom neuronunit.optimisation.optimization_management import dtc_to_rheo, switch_logic,active_values\nnu_tests = switch_logic(nu_tests)\nfrom neuronunit.tests.base import AMPL, DELAY, DURATION\n\ndef rheobase_protocol(model,nu_tests):\n    dtc = model.model_to_dtc()\n    dtc = dtc_to_rheo(dtc)\n    model.rheobase = dtc.rheobase\n    for v in nu_tests:\n        k = v.name\n        if not hasattr(v,'params'):\n            v.params = {}\n        if not 'injected_square_current' in v.params.keys():    \n            v.params['injected_square_current'] = {}\n        if v.passive == False and v.active == True:\n            keyed = v.params['injected_square_current']\n            v.params = active_values(keyed,dtc.rheobase)\n            v.params['injected_square_current']['delay'] = DELAY\n            v.params['injected_square_current']['duration'] = DURATION\n    return model,nu_tests\n\n\ndef evaluate(nu_tests,model):\n    scores_ = []\n    suite = TestSuite(nu_tests)\n    for t in suite:\n        if 'RheobaseTest' in t.name: t.score_type = sciunit.scores.ZScore\n        if 'RheobaseTestP' in t.name: t.score_type = sciunit.scores.ZScore\n        if 'mean' not in t.observation.keys():\n            t.observation['mean'] = t.observation['value']\n        score_gene = t.judge(model)\n        if not isinstance(type(score_gene),type(None)):\n            if not isinstance(type(score_gene),sciunit.scores.InsufficientDataScore):\n                if not isinstance(type(score_gene.log_norm_score),type(None)):\n                    try:\n\n                        lns = np.abs(score_gene.log_norm_score)\n                    except:\n                        # works 1/2 time that log_norm_score does not work\n                        # more informative than nominal bad score 100\n                        lns = np.abs(score_gene.raw)\n                else:\n                    # works 1/2 time that log_norm_score does not work\n                    # more informative than nominal bad score 100\n\n                    lns = np.abs(score_gene.raw)\n            else:\n                pass\n        scores_.append(lns)\n    for i,s in enumerate(scores_):\n        if s==np.inf:\n            scores_[i] = 100 #np.abs(float(score_gene.raw))\n    SA = ScoreArray(nu_tests, scores_)\n    fitness = (v for v in SA.values)\n    objectives = SA.to_dict()\n    return fitness\n'''\n\n#ephys.objectivescalculators.ObjectivesCalculator = \n#score_calc = ephys.objectivescalculators.ObjectivesCalculator(nu_tests,model) \n#score_calc.objectives = objectives")


# In[ ]:



lop={}
from bluepyopt.parameters import Parameter
for k,v in MODEL_PARAMS["IZHI"].items():
    p = Parameter(name=k,bounds=v,frozen=False)
    lop[k] = p
    
simple_cell.params = lop

nu_tests[0].judge(simple_cell)
from sciunit.scores import ZScore
nu_tests[0].score_type = ZScore


# Combining everything together we have a CellEvaluator. The CellEvaluator constructor has a field 'parameter_names' which contains the (ordered) list of names of the parameters that are used as input (and will be fitted later on).

# In[ ]:


MODEL_PARAMS["IZHI"]
cell_evaluator = ephys.evaluators.CellEvaluator(
        cell_model=simple_cell,
        param_names=MODEL_PARAMS["IZHI"].keys(),
        fitness_protocols={twostep_protocol.name: twostep_protocol},
        fitness_calculator=score_calc,
        sim='euler')
simple_cell.params_by_names(MODEL_PARAMS["IZHI"].keys())
simple_cell.params;


# ### Evaluating the cell
# 
# The cell can now be evaluate for a certain set of parameter values.

# In[ ]:


#default_params = MODEL_PARAMS["IZHI"]
#print(cell_evaluator.evaluate_with_dicts())


# ## Setting up and running an optimisation
# 
# Now that we have a cell template and an evaluator for this cell, we can set up an optimisation.

# In[ ]:


optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator,
        offspring_size = 10)


# And this optimisation can be run for a certain number of generations

# In[ ]:


final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=35)


# The optimisation has return us 4 objects: final population, hall of fame, statistical logs and history. 
# 
# The final population contains a list of tuples, with each tuple representing the two parameters of the model

# In[ ]:


print('Final population: ', final_pop)


# The best individual found during the optimisation is the first individual of the hall of fame

# In[ ]:


best_ind = hall_of_fame[0]
print('Best individual: ', best_ind)
print('Fitness values: ', best_ind.fitness.values)


# We can evaluate this individual and make use of a convenience function of the cell evaluator to return us a dict of the parameters

# In[ ]:


best_ind_dict = cell_evaluator.param_dict(best_ind)
print(cell_evaluator.evaluate_with_dicts(best_ind_dict))


# In[ ]:


model = cell_evaluator.cell_model
cell_evaluator.param_dict(best_ind)
model.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}


# In[ ]:



opt = model.model_to_dtc()
opt.attrs = {str(k):float(v) for k,v in cell_evaluator.param_dict(best_ind).items()}

check_binary_match(target,opt)
inject_and_plot_passive_model(opt,second=target)


# As you can see the evaluation returns the same values as the fitness values provided by the optimisation output. 
# We can have a look at the responses now.

# In[ ]:


#plot_responses(twostep_protocol.run(cell_model=simple_cell, param_values=best_ind_dict, sim=nrn))
 


# Let's have a look at the optimisation statistics.
# We can plot the minimal score (sum of all objective scores) found in every optimisation. 
# The optimisation algorithm uses negative fitness scores, so we actually have to look at the maximum values log.

# In[ ]:


import numpy
gen_numbers = logs.select('gen')
min_fitness = logs.select('min')
max_fitness = logs.select('max')
plt.plot(gen_numbers, min_fitness, label='min fitness')
plt.xlabel('generation #')
plt.ylabel('score (# std)')
plt.legend()
plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1) 
plt.ylim(0.9*min(min_fitness), 1.1 * max(min_fitness)) 


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

from bluepyopt.allenapi.allen_data_driven import opt_setup, opt_setup_two, opt_exec, opt_to_model
from neuronunit.optimisation.optimization_management import check_bin_vm15
from neuronunit.optimisation.model_parameters import MODEL_PARAMS, BPO_PARAMS, to_bpo_param
from neuronunit.optimisation.optimization_management import dtc_to_rheo,inject_and_plot_model
from bluepyopt.allenapi.allen_data_driven import opt_to_model
from bluepyopt.allenapi.utils import dask_map_function
import matplotlib.pyplot as plt
import numpy as np
from neuronunit.optimisation.data_transport_container import DataTC
import efel
from jithub.models import model_classes

import quantities as qt


ids = [ 324257146,
        325479788,
        476053392,
        623893177,
        623960880,
        482493761,
        471819401
       ]

specimen_id = ids[1]
efel.__file__
efel_list = list(efel.getFeatureNames());
cellmodel = "ADEXP"


# # TODO make a nested Genetic Algorithm where the outer loop explores different preferred currents.
#
# This will get rid of the oscillations.
if cellmodel == "IZHI":
    model = model_classes.IzhiModel()
if cellmodel == "MAT":
    model = model_classes.MATModel()
if cellmodel == "ADEXP":
    model = model_classes.ADEXPModel()


# specimen id 623960880 \\
# {\small
# \url{http://celltypes.brain-map.org/mouse/experiment/electrophysiology/623960880}
# specimen id 623893177 \\
# \url{http://celltypes.brain-map.org/mouse/experiment/electrophysiology/623893177}
# specimen id 482493761 \\
# \url{http://celltypes.brain-map.org/mouse/experiment/electrophysiology/482493761}
# specimen id 471819401 \\
# \url{http://celltypes.brain-map.org/mouse/experiment/electrophysiology/471819401}
#

# In[3]:


'AHP_depth_abs_1.5x',
'sag_ratio2_1.5x',
'ohmic_input_resistance_1.5x',
'sag_ratio2_1.5x',
'peak_voltage_1.5x',
'voltage_base_1.5x',
'voltage',


# In[4]:


specific_filter_list = ['ISI_log_slope_1.5x',
                        'mean_frequency_1.5x',
                        'adaptation_index2_1.5x',
                        'first_isi_1.5x',
                        'ISI_CV_1.5x',
                        'median_isi_1.5x',
                        'Spikecount_1.5x',
                        'all_ISI_values',
                        'ISI_values',
                        'time_to_first_spike',
                        'time_to_last_spike',
                        'time_to_second_spike',
                        'spike_times']
simple_yes_list = specific_filter_list
target_num_spikes = 8


# In[5]:


dtc = DataTC()
dtc.backend = cellmodel
dtc._backend = model._backend

dtc.attrs = model.attrs
dtc.params = {k:np.mean(v) for k,v in MODEL_PARAMS[cellmodel].items()}

model = dtc.dtc_to_model()




dtc = dtc_to_rheo(dtc)

vm,plt,dtc = inject_and_plot_model(dtc,plotly=False)

fixed_current = 122 *qt.pA
model.params

model, suite, nu_tests, target_current, spk_count = opt_setup(specimen_id,
                                                              cellmodel,
                                                              target_num_spikes,provided_model=model,fixed_current=False)





model.seeded_current = target_current['value']
model.allen = True
model.seeded_current
model.NU = True
cell_evaluator,simple_cell = opt_setup_two(model,cellmodel, suite, nu_tests, target_current, spk_count,provided_model=model)
#mat.NU = True
NGEN = 100
MU = 15

mapping_funct = dask_map_function
final_pop, hall_of_fame, logs, hist = opt_exec(MU,NGEN,mapping_funct,cell_evaluator)


opt,target = opt_to_model(hall_of_fame,cell_evaluator,suite, target_current, spk_count)


best_ind = hall_of_fame[0]
fitnesses = cell_evaluator.evaluate_with_lists(best_ind)


obnames = [obj.name for obj in cell_evaluator.objectives]

for i,j in zip(fitnesses,obnames):
    print(i,j)
    assert i<0.07

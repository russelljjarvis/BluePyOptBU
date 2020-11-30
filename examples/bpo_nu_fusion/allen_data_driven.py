import pickle
import make_allen_tests_from_id# import *

from make_allen_tests_from_id import *
from neuronunit.optimisation.optimization_management import dtc_to_rheo
from neuronunit.optimisation.optimization_management import inject_model30,check_bin_vm30,check_bin_vm15


import efel
import pandas as pd
import seaborn as sns
list(efel.getFeatureNames());
from utils import dask_map_function

import bluepyopt as bpop
import bluepyopt.ephys as ephys
import pickle
from sciunit.scores import ZScore, RatioScore
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
import numpy
from neuronunit.optimisation.optimization_management import test_all_objective_test
from neuronunit.optimisation.optimization_management import check_binary_match, three_step_protocol,inject_and_plot_passive_model
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
import copy

import numpy as np
from make_allen_tests import AllenTest

from sciunit.scores import ZScore
from collections.abc import Iterable

from bluepyopt.parameters import Parameter
from utils import dask_map_function


#tests = pickle.load(open('allen_NU_tests.p','rb'))
#names = [t.name for t in tests[3].tests ]
#names;


simple_yes_list = [
    'all_ISI_values',
    'ISI_log_slope','mean_frequency',
    'adaptation_index2',
    'first_isi','ISI_CV','median_isi','AHP_depth_abs',
    'sag_ratio2','ohmic_input_resistance',
    'sag_ratio2','peak_voltage','voltage_base','Spikecount',
    'ohmic_input_resistance_vb_ssse',
    'all_ISI_values',
    'ISI_values',
    'time_to_first_spike',
    'time_to_last_spike',
    'time_to_second_spike',
    'trace_check']

#simple_yes_list = ['mean_frequency','ISI_log_slope','adaptation_index2','AHP_depth_abs','sag_ratio2','ohmic_input_resistance','sag_ratio2','peak_voltage','voltage_base','Spikecount','ohmic_input_resistance_vb_ssse']


#names = [t.observation for t in tests[3].tests if "Spike" in t.name]
#names
#specimen_id = tests[3].name
# initialize the cacher

def opt_setup(specimen_id,cellmodel,target_num):
    try:
        assert 1==2
        with open(str(specimen_id)+'later_allen_NU_tests.p','rb') as f:
            suite = pickle.load(f)

    except:
        #specimen_id = tests[4].name
        sweep_numbers,data_set,sweeps = make_allen_tests_from_id.allen_id_to_sweeps(specimen_id)
        target_num = 20
        for i in range(0,target_num):
            #                                                    def get_model_parts_sweep_from_spk_cnt(spk_cnt,data_set,sweep_numbers,specimen_id):

            vmm,stimulus,sn,spike_times = make_allen_tests_from_id.get_model_parts_sweep_from_spk_cnt(20,data_set,sweep_numbers,specimen_id)
            if spike_times is not None:
                if len(spike_times) != 0:
                    print(vmm)
                    break
        #vm15,vm30,rheobase,currents,vmrh = make_allen_tests_from_id.get_model_parts(data_set,sweep_numbers,specimen_id)
        #print(vmm)
        #import pdb
        #pdb.set_trace()
        suite,specimen_id = make_allen_tests_from_id.make_suite_known_sweep_from_static_models(vmm,stimulus,specimen_id)

        #suite,specimen_id = make_allen_tests_from_id.make_suite_from_static_models(vm15,vm30,rheobase,currents,vmrh,specimen_id)
        with open(str(specimen_id)+'later_allen_NU_tests.p','wb') as f:
            pickle.dump(suite,f)

    #specific_filter_list = ['all_ISI_values_3.0x','ISI_log_slope_3.0x','mean_frequency_3.0x','adaptation_index2_3.0x','first_isi_3.0x','ISI_CV_3.0x','median_isi_3.0x','AHP_depth_abs_3.0x','sag_ratio2_3.0x','ohmic_input_resistance_3.0x','sag_ratio2_3.0x','peak_voltage_3.0x','voltage_base_3.0x','Spikecount_3.0x','ohmic_input_resistance_vb_ssse_3.0x']


    target = StaticModel(vm=suite.traces['vmrh']) #DataTC(backend="IZHI")
    #target.vm30 = suite.traces['vm30']
    target.vm15 = suite.traces['vm15']

    nu_tests = suite.tests;
    check_bin_vm15(target,target)


    attrs = {k:np.mean(v) for k,v in MODEL_PARAMS[cellmodel].items()}
    dtc = DataTC(backend=cellmodel,attrs=attrs)
    for t in nu_tests:
        #print(t.name)
        if t.name == 'Spikecount_1.5x':
            spk_count = float(t.observation['mean'])
            #print(spk_count,'spike_count')
            break
    observation_range={}
    observation_range['value'] = spk_count
    scs = SpikeCountSearch(observation_range)
    model = dtc.dtc_to_model()
    target_current = scs.generate_prediction(model)
    #model.inject_square_current()

    ALLEN_DELAY = 1000.0*qt.s
    ALLEN_DURATION = 2000.0*qt.s


    uc = {'amplitude':target_current['value'],'duration':ALLEN_DURATION,'delay':ALLEN_DELAY}
    model = dtc.dtc_to_model()

    model.inject_square_current(uc)
    #vm15 = model.get_membrane_potential()
    #print(model.get_spike_count(),'spikes')


    tg = target_current['value']
    MODEL_PARAMS[cellmodel]#["current_inj"] = [tg-0.25*tg,tg+0.25*tg]

    simple_cell = ephys.models.ReducedCellModel(
            name='simple_cell',
            params=MODEL_PARAMS[cellmodel],backend=cellmodel)
    simple_cell.backend = cellmodel
    simple_cell.allen = None
    simple_cell.allen = True


    model = simple_cell
    model.params = {k:np.mean(v) for k,v in model.params.items() }

    features = None
    allen = True



    return model, suite, nu_tests, target_current, spk_count

class NUFeatureAllenMultiSpike(object):
    def __init__(self,test,model,cnt,target,spike_obs,print_stuff=False):
        self.test = test
        self.model = model
        #self.check_list = check_list
        self.spike_obs = spike_obs
        self.cnt = cnt
        self.target = target
        self.print_stuff = print_stuff
    def calculate_score(self,responses):
        #print(responses.keys())
        #import pdb
        #pdb.set_trace()
        if not 'features' in responses.keys():# or not 'model' in responses.keys():
            return 1000.0
        features = responses['features']
        if features is None:
            return 1000.0
        #check_list = self.check_list

        self.test.score_type = RatioScore

        feature_name = self.test.name
        #print(features.keys())
        #delta1 = np.abs(features['Spikecount_1.5x']-np.mean(self.spike_obs[0]['mean']))
        if feature_name not in features.keys():
            return 1000.0#+(delta1)

        if features[feature_name] is None:
            return 1000.0#+(delta1)

        if type(features[self.test.name]) is type(Iterable):
            features[self.test.name] = np.mean(features[self.test.name])
        self.test.observation['std'] = np.abs(np.mean(self.test.observation['mean']))
        self.test.observation['mean'] = np.mean(self.test.observation['mean'])
        self.test.set_prediction(np.mean(features[self.test.name]))

        #if 'Spikecount_3.0x'==feature_name or
        if 'Spikecount_1.5x'==feature_name:
            delta = np.abs(features[self.test.name]-np.mean(self.test.observation['mean']))
            if np.nan==delta or delta==np.inf:
                delta = 1000.0


            return delta
        else:


            #if feature_name in check_list:
            if features[feature_name] is None:
                print('gets here')
                return 1000.0+(delta1)
            self.test.score_type = ZScore
            score_gene = self.test.feature_judge()
            #print(score_gene)
            #import pdb
            #pdb.set_trace()
            if score_gene is not None:
                if score_gene.log_norm_score is not None:
                    delta = np.abs(float(score_gene.log_norm_score))
                else:
                    if score_gene.raw is not None:
                        delta = np.abs(float(score_gene.raw))
                    else:
                        delta = None

            else:
                delta = None
                    #if delta==np.inf or np.isnan(delta):
                    #    if score_gene.raw is not None:
                    #        delta =  np.abs(float(score_gene.raw))
            if delta is None:
                delta = np.abs(features[self.test.name]-np.mean(self.test.observation['mean']))


            if np.nan==delta or delta==np.inf:
                delta = np.abs(features[self.test.name]-np.mean(self.test.observation['mean']))
            if np.nan==delta or delta==np.inf:
                delta = 1000.0

            return delta#+delta2
            #else:
            #    return 1000.0


def opt_setup_two(model, cellmodel, suite, nu_tests, target_current, spk_count):
    objectives = []
    spike_obs = []
    for tt in nu_tests:
        #if 'Spikecount_3.0x' == tt.name:
        #    spike_obs.append(tt.observation)
        if 'Spikecount_1.5x' == tt.name:
            #print(tt.observation)
            spike_obs.append(tt.observation)
    spike_obs = sorted(spike_obs, key=lambda k: k['mean'],reverse=True)

    #check_list["RheobaseTest"] = target.rheobase['value']
    for cnt,tt in enumerate(nu_tests):
        feature_name = '%s' % (tt.name)
        #if feature_name in specific_filter_list:
            #if 'Spikecount_3.0x' == tt.name or
        if 'Spikecount_1.5x' == tt.name:
            #def __init__(self,test,model,cnt,target,spike_obs,print_stuff=False):

            ft = NUFeatureAllenMultiSpike(tt,model,cnt,target_current,spike_obs)
            objective = ephys.objectives.SingletonObjective(
                feature_name,
                ft)
            objectives.append(objective)


    score_calc = ephys.objectivescalculators.ObjectivesCalculator(objectives)

    lop={}

    for k,v in MODEL_PARAMS[cellmodel].items():
        p = Parameter(name=k,bounds=v,frozen=False)
        lop[k] = p


    simple_cell = ephys.models.ReducedCellModel(
            name='simple_cell',
            params=MODEL_PARAMS[cellmodel],backend=cellmodel)
    simple_cell.backend = cellmodel
    simple_cell.params = lop
    sweep_protocols = []
    for protocol_name, amplitude in [('step1', 0.05)]:

        protocol = ephys.protocols.SweepProtocol(protocol_name, [None], [None])
        sweep_protocols.append(protocol)
    twostep_protocol = ephys.protocols.SequenceProtocol('twostep', protocols=sweep_protocols)

    simple_cell.params_by_names(MODEL_PARAMS[cellmodel].keys())
    simple_cell.params;

    MODEL_PARAMS[cellmodel]
    cell_evaluator = ephys.evaluators.CellEvaluator(
            cell_model=simple_cell,
            param_names=MODEL_PARAMS[cellmodel].keys(),
            fitness_protocols={twostep_protocol.name: twostep_protocol},
            fitness_calculator=score_calc,
            sim='euler')

    simple_cell.params_by_names(MODEL_PARAMS[cellmodel].keys())
    simple_cell.params;
    simple_cell.seeded_current = target_current['value']
    simple_cell.spk_count = spk_count



    #no_list = pickle.load(open("too_rippled_b.p","rb"))


    objectives2 = []
    for cnt,tt in enumerate(nu_tests):
        feature_name = '%s' % (tt.name)
        #if (feature_name in specific_filter_list):
        #if feature_name != "time_constant_1.5x" and feature_name != "RheobaseTest":
        ft = NUFeatureAllenMultiSpike(tt,model,cnt,target_current,spike_obs,print_stuff=True)
        #ft = NUFeatureAllenMultiSpike(tt,model,cnt,target_current,spike_obs)

        objective = ephys.objectives.SingletonObjective(
            feature_name,
            ft)
        objectives2.append(objective)
    objectives2
    score_calc2 = ephys.objectivescalculators.ObjectivesCalculator(objectives2)
    objectives2


    simple_cell.params_by_names(MODEL_PARAMS[cellmodel].keys())
    simple_cell.params;


    MODEL_PARAMS[cellmodel]
    cell_evaluator2 = ephys.evaluators.CellEvaluator(
            cell_model=simple_cell,
            param_names=list(MODEL_PARAMS[cellmodel].keys()),
            fitness_protocols={twostep_protocol.name: twostep_protocol},
            fitness_calculator=score_calc2,
            sim='euler')
    return cell_evaluator2,simple_cell


MU = 20
def opt_exec(MU,NGEN,mapping_funct,cell_evaluator2):
    optimisation = bpop.optimisations.DEAPOptimisation(
            evaluator=cell_evaluator2,
            offspring_size = MU,
            map_function = map,
            selector_name='IBEA',mutpb=0.1,cxpb=0.35)
    final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=NGEN)
    return final_pop, hall_of_fame, logs, hist

def opt_to_model(hall_of_fame,cell_evaluator2,suite, target_current, spk_count):
    best_ind = hall_of_fame[0]
    best_ind_dict = cell_evaluator2.param_dict(best_ind)
    model = cell_evaluator2.cell_model
    cell_evaluator2.param_dict(best_ind)

    model.attrs = {str(k):float(v) for k,v in cell_evaluator2.param_dict(best_ind).items()}
    model._backend.attrs = model.attrs

    opt = model.model_to_dtc()
    opt.attrs = {str(k):float(v) for k,v in cell_evaluator2.param_dict(best_ind).items()}
    model._backend.attrs = opt.attrs
    target = copy.copy(opt)
    #target.vm30 = suite.traces['vm30']
    target.vm15 = suite.traces['vm15']
    #opt.allen = None
    #opt.allen = True
    opt.seeded_current = target_current['value']
    opt.spk_count = spk_count

    target.seeded_current = target_current['value']
    target.spk_count = spk_count


    vm301,vm151,_,target = inject_model30(target,known_current=target_current['value'])
    vm302,vm152,_,opt = inject_model30(opt,known_current=target_current['value'])
    return opt,target
    '''
    #check_bin_vm30(opt,opt)
    check_bin_vm15(opt,opt)


    #check_bin_vm30(target,target)


    check_bin_vm15(target,target)



    gen_numbers = logs.select('gen')
    min_fitness = logs.select('min')
    max_fitness = logs.select('max')
    avg_fitness = logs.select('avg')
    plt.plot(gen_numbers, max_fitness, label='max fitness')
    plt.plot(gen_numbers, avg_fitness, label='avg fitness')
    plt.plot(gen_numbers, min_fitness, label='min fitness')

    plt.plot(gen_numbers, min_fitness, label='min fitness')
    plt.semilogy()
    plt.xlabel('generation #')
    plt.ylabel('score (# std)')
    plt.legend()
    plt.xlim(min(gen_numbers) - 1, max(gen_numbers) + 1)
    plt.show()
    '''
'''
cp = {}
cp['final_pop'] = final_pop
cp['hall_of_fame'] = hall_of_fame


#with open('allen_opt.p','wb') as f:
#    pickle.dump(f,[final_pop, hall_of_fame, logs, hist])









optimisation = bpop.optimisations.DEAPOptimisation(
        evaluator=cell_evaluator2,
        offspring_size = MU,
        map_function = dask_map_function,
        selector_name='IBEA',mutpb=0.1,cxpb=0.35,seeded_pop=[cp['final_pop'],cp['hall_of_fame']])#,seeded_current=target_current)
final_pop, hall_of_fame, logs, hist = optimisation.run(max_ngen=50)
'''

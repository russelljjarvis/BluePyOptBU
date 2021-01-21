"""
BPO front end for reduced neural models
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go # or plotly.express as px
import sklearn
import os
# sense if running on heroku
if 'DYNO' in os.environ:
    heroku = False
else:
    heroku = True
import pickle
experimental_constraints = pickle.load(open("../data_driven/processed_multicellular_constraints.p","rb"))
olfactory_bulb_constraints = pickle.load(open("olf_tests.p","rb"))

import utils
import streamlit as st
import bluepyopt as bpop
from neuronunit.optimisation.model_parameters import MODEL_PARAMS
from neuronunit.optimisation.optimization_management import inject_and_plot_model, inject_and_plot_passive_model, dtc_to_rheo
#from neuronunit.optimisation.optimization_management import inject_and_plot_model, inject_and_plot_passive_model, dtc_to_rheo


import pandas as pd

from neuronunit.optimisation.optimization_management import TSD, display_fitting_data

from neuronunit.optimisation.data_transport_container import DataTC
import matplotlib.pyplot as plt
from neuronunit.capabilities.spike_functions import get_spike_waveforms
from quantities import ms
from neuronunit.tests.base import AMPL, DELAY, DURATION
MODEL_PARAMS['NEURONHH'] = { k:sorted(v) for k,v in MODEL_PARAMS['NEURONHH'].items() }

import os
import base64
import seaborn as sns
from scipy.stats import norm

#from utils import plot_as_normal
from neuronunit.optimisation.optimization_management import instance_opt
from neuronunit.optimisation.optimization_management import plot_as_normal
from neuronunit.tests import *
import quantities as qt
from sciunit import TestSuite
from sciunit.scores import ZScore, RatioScore

def make_allen():

  rt = RheobaseTest(observation={'mean':70*qt.pA,'std':70*qt.pA})
  tc = TimeConstantTest(observation={'mean':24.4*qt.ms,'std':24.4*qt.ms})
  ir = InputResistanceTest(observation={'mean':132*qt.MOhm,'std':132*qt.MOhm})
  rp = RestingPotentialTest(observation={'mean':-71.6*qt.mV,'std':77.5*qt.mV})

  allen_tests = [rt,tc,rp,ir]
  for t in allen_tests:
      t.score_type = RatioScore
  allen_tests[-1].score_type = ZScore
  allen_suite482493761 = TestSuite(allen_tests)
  allen_suite482493761.name = "http://celltypes.brain-map.org/mouse/experiment/electrophysiology/482493761"

  rt = RheobaseTest(observation={'mean':190*qt.pA,'std':190*qt.pA})
  tc = TimeConstantTest(observation={'mean':13.8*qt.ms,'std':13.8*qt.ms})
  ir = InputResistanceTest(observation={'mean':132*qt.MOhm,'std':132*qt.MOhm})
  rp = RestingPotentialTest(observation={'mean':-77.5*qt.mV,'std':77.5*qt.mV})

  allen_tests = [rt,tc,rp,ir]
  for t in allen_tests:
      t.score_type = RatioScore
  allen_tests[-1].score_type = ZScore
  allen_suite471819401 = TestSuite(allen_tests)
  allen_suite471819401.name = "http://celltypes.brain-map.org/mouse/experiment/electrophysiology/471819401"
  list_of_dicts = []
  cells={}
  cells['471819401'] = TSD(allen_suite471819401)
  cells['482493761'] = TSD(allen_suite482493761)

  for k,v in cells.items():
      observations = {}
      for k1 in cells['482493761'].keys():
          vsd = TSD(v)
          if k1 in vsd.keys():
              vsd[k1].observation['mean']

              observations[k1] = np.round(vsd[k1].observation['mean'],2)
              observations['name'] = k
      list_of_dicts.append(observations)
  df = pd.DataFrame(list_of_dicts)
  df


  return allen_suite471819401,allen_suite482493761,df

def get_binary_file_downloader_html(bin_file_path, file_label='File'):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file_path)}">Download {file_label}</a>'
    return href

def color_large_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = "red" if val > 2.0 else "white"
    return "color: %s" % color


def highlight_min(data, color="yellow"):
    """highlight the maximum in a Series or DataFrame"""
    attr = "background-color: {}".format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_max = data == data.min()
        return [attr if v else "" for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.min().min()
        return pd.DataFrame(
            np.where(is_max, attr, ""), index=data.index, columns=data.columns
        )


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}



def plot_imshow_plotly(df):

    heat = go.Heatmap(df_to_plotly(df))
    #fig = go.Figure(data=

    title = 'Lognorm Score Matrix NeuronUnit'

    layout = go.Layout(title_text=title, title_x=0.5,
                    width=600, height=600,
                    xaxis_showgrid=True,
                    yaxis_showgrid=True)

    fig=go.Figure(data=[heat], layout=layout)

    st.write(fig)


if __name__ == "__main__":
    allen_suite471819401,allen_suite482493761,df = make_allen_tests()
    st.title('Reduced Model Fitting to Neuroelectro Experimental Constraints')
    st.markdown('------')
    st.markdown('Select the measurements you want to use to guide fitting')



    experimental_constraints.pop("Olfactory bulb (main) mitral cell")
    olf_bulb = {'mitral olfactory bulb cell':olfactory_bulb_constraints}
    experimental_constraints.update(olf_bulb)
    experimental_constraints["Allen471819401"] = allen_suite471819401
    experimental_constraints["Allen482493761"] = allen_suite482493761

    test_key = st.sidebar.radio("\
      What experiments would you like to fit models to?"
		,tuple(experimental_constraints.keys()))
    view_data_in_detail = st.sidebar.radio("\
      view data in detail?"
      ,("No","Yes"))
    if view_data_in_detail=="Yes":
      df = display_fitting_data()
      st.table(df)

    experimental_constraints = TSD(experimental_constraints[test_key])

    test_keys = list(experimental_constraints.keys())
    subset_tests = st.sidebar.radio("\
		Would you like to fit models on a subset of tests?"
		,("No","Yes"))
    if subset_tests == "Yes":
      test_keys = st.sidebar.multiselect("\
      Are you interested in less than all of the features?"
      ,tuple(experimental_constraints.keys()))
    else:
      test_keys = [k for k in experimental_constraints.keys() if k not in set(["InjectedCurrentAPThresholdTest","InjectedCurrentAPWidthTest","InjectedCurrentAPAmplitudeTest"])]


    full_test_list = copy.copy(experimental_constraints)

    experimental_constraints = [ experimental_constraints[k] for k in test_keys ]
    model_value = st.sidebar.radio("\
		Which models would you like to optimize"
		,("ADEXP","IZHI","NEURONHH"))

    diversity = st.sidebar.radio("\
		Do you want diverse solutions or just the best solution?"
		,("IBEA","NSGA2"))

    readiness = st.radio("\
		Ready to go?"
		,("No","Yes"))

    MU = st.sidebar.radio("\
		Population size is"
		,(25,50,75,100))
    NGEN = st.sidebar.radio("\
		Number of generations is"
		,(50,75,100,125))

    if readiness == "Yes":
      instance_opt(experimental_constraints,MODEL_PARAMS,test_key,model_value,MU,NGEN,diversity,full_test_list,use_streamlit=True)


      if model_value == "ADEXP":
        st.markdown('''
        In the {2} Model:
        Over three different current injection strengths mean Model speed was {0}
        ms, to find unkown rheoase value it takes {1} seconds'''.format(300,3.8,model_value))


      if model_value == "IZHI":
        st.markdown('''
        The IZHI model is the fastest.
        Over three different current injection strengths mean Model speed was {0}
        ms, to find unkown rheoase value it takes {1} seconds'''.format(3.36,0.7,model_value))


      if model_value == "NEURONHH":
        st.markdown('''
        In the {2} Model:
        over three different current injection strengths mean Model speed was {0}
        ms, to find unkown rheoase value it takes {1} seconds'''.format(3.36,0.7,model_value))



      '''
      # Do you want to try again on a different model to see how that would look?
      '''

      another_go = st.radio("\
		  Want to keep model on screen and compare to new model?"
		  ,("No","Yes"))

      new_model_value = st.sidebar.radio("\
      Which models would you like to optimize"
      ,("ADEXP","IZHI","NEURONHH"))
      if another_go == "Yes" and (new_model_value is not model_value):
        instance_opt(experimental_constraints,MODEL_PARAMS,test_key,model_value,MU,NGEN,diversity,full_test_list,use_streamlit=True)

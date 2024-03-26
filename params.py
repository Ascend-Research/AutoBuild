import os
from os.path import sep as P_SEP


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = P_SEP.join([BASE_DIR, "cache"])
SAVED_MODELS_DIR = P_SEP.join([BASE_DIR, "saved_models"])
CONFIG_DIR = P_SEP.join([BASE_DIR, "configs"])
UNITS_DIR = P_SEP.join([BASE_DIR, "units"])
EVALS_DIR = P_SEP.join([BASE_DIR, "evals"])
PLOTS_DIR = P_SEP.join([BASE_DIR, "plots"])


if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
if not os.path.exists(SAVED_MODELS_DIR): os.makedirs(SAVED_MODELS_DIR)
if not os.path.exists(CONFIG_DIR): os.makedirs(CONFIG_DIR)
if not os.path.exists(UNITS_DIR): os.makedirs(UNITS_DIR)
if not os.path.exists(EVALS_DIR): os.makedirs(EVALS_DIR)
if not os.path.exists(PLOTS_DIR): os.makedirs(PLOTS_DIR)

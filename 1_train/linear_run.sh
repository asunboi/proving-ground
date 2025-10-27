#!/bin/bash

HYDRA_FULL_ERROR=1 train experiment=state_linear_baseline hpo=linear_additive_hpo
HYDRA_FULL_ERROR=1 predict experiment=predict_state_linear_baseline
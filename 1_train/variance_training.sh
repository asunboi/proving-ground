#!/bin/bash

HYDRA_FULL_ERROR=1 train -m '+experiment/boli_251006_1=glob(*)'
HYDRA_FULL_ERROR=1 train -m '+experiment/boli_251006_2=glob(*)'
HYDRA_FULL_ERROR=1 train -m '+experiment/boli_251006_3=glob(*)'
HYDRA_FULL_ERROR=1 train -m '+experiment/boli_251006_4=glob(*)'
HYDRA_FULL_ERROR=1 train -m '+experiment/boli_251006_5=glob(*)'

#!/bin/bash
GPU=$1
SEED=$2
ENV=$3
EPOCHS=$4
ALG1=$5
N_RUNS=$6
UNDERLINE="_"
EXP="experiments"
if [ $# == 6 ];then
  TITLE="clean"
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_dqn.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --dir $DIR
  fi
  exit 0
fi

C=$7
B1=$8
B2=$9
TYPE=${10}

if [ $TYPE == uniform ];then
  TITLE=uniform$UNDERLINE$C$UNDERLINE$B1$UNDERLINE$B2
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_dqn.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --atk_params $C $B1 $B2 --uniform --dir $DIR
  fi
elif [ $TYPE == target ];then
  ALG2=${11}
  PFM=${12}
  TITLE=target$UNDERLINE$C$UNDERLINE$B1$UNDERLINE$B2
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_dqn.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --atk_params $C $B1 $B2 --target1 --atk_alg $ALG2 --atk_pfm $PFM --dir $DIR
  fi
elif [ $TYPE == promote ];then
  ALG2=${11}
  PFM=${12}
  TITLE=promote$UNDERLINE$C$UNDERLINE$B1$UNDERLINE$B2
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_dqn.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --atk_params $C $B1 $B2 --promote --atk_alg $ALG2 --atk_pfm $PFM --dir $DIR
  fi
elif [ $TYPE == online ];then
  ALG2=${11}
  CHANGE=${12}
  TITLE=online$UNDERLINE$C$UNDERLINE$B1$UNDERLINE$B2
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_dqn.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --atk_params $C $B1 $B2 --atk_online --atk_alg $ALG2 --dir $DIR --change $CHANGE
  fi
else
  ALG2=${11}
  PFM=${12}
  TITLE=offline$UNDERLINE$PFM$UNDERLINE$C$UNDERLINE$B1$UNDERLINE$B2
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_dqn.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --atk_params $C $B1 $B2 --atk_alg $ALG2 --atk_pfm $PFM --dir $DIR
  fi
fi

#!/bin/bash
GPU=$1
SEED=$2
ENV=$3
EPOCHS=$4
ALG1=$5
N_RUNS=$6
UNDERLINE="_"
atk_radius=2
EXP="experiments"
if [ $ENV == Walker2d-v2 ];then
  atk_radius=2.2
fi
if [ $ENV == Swimmer-v3 ];then
  atk_radius=2.2
fi
if [ $# == 6 ];then
  TITLE="clean"
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_class.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --dir $DIR
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
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_class.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --atk_params $C $B1 $B2 --uniform --dir $DIR --atk_radius $atk_radius
  fi
elif [ $TYPE == target ];then
  ALG2=${11}
  TITLE=target$UNDERLINE$C$UNDERLINE$B1$UNDERLINE$B2$UNDERLINE$atk_radius
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ $ENV == Walker2d-v2 ];then
    atk_radius=1.0
  elif [ $ENV == Swimmer-v3 ];then
    atk_radius=1.5
  elif [ $ENV == Hopper-v3 ];then
    atk_radius=1.1
  elif [ $ENV == HalfCheetah-v2 ];then
    atk_radius=1.5
  fi
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_class.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --atk_params $C $B1 $B2 --target --atk_alg $ALG2 --atk_pfm -2000 --dir $DIR  --atk_radius $atk_radius
  fi
elif [ $TYPE == promote ];then
  ALG2=${11}
  TITLE=promote$UNDERLINE$C$UNDERLINE$B1$UNDERLINE$B2$UNDERLINE$atk_radius
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ $ENV == Walker2d-v2 ];then
    atk_radius=1.0
  elif [ $ENV == Swimmer-v3 ];then
    atk_radius=1.5
  elif [ $ENV == Hopper-v3 ];then
    atk_radius=1.1
  elif [ $ENV == HalfCheetah-v2 ];then
    atk_radius=1.5
  fi
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_class.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --atk_params $C $B1 $B2 --promote --atk_alg $ALG2 --atk_pfm -2000 --dir $DIR  --atk_radius $atk_radius
  fi
elif [ $TYPE == online ];then
  ALG2=${11}
  TITLE=online$UNDERLINE$C$UNDERLINE$B1$UNDERLINE$B2
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_class.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --atk_params $C $B1 $B2 --atk_online --atk_alg $ALG2 --dir $DIR  --atk_radius $atk_radius
  fi
elif [ $TYPE == offline ];then
  ALG2=${11}
  PFM=${12}
  TITLE=offline$UNDERLINE$PFM$UNDERLINE$C$UNDERLINE$B1$UNDERLINE$B2$UNDERLINE$atk_radius
  EXP_NAME=$ENV$UNDERLINE$ALG1$UNDERLINE$TITLE
  DIR=../$EXP/$ENV$UNDERLINE$SEED/$EXP_NAME
  if [ ! -d $DIR ];then
    mkdir $DIR
    CUDA_VISIBLE_DEVICES=$GPU python ../src/all_class.py --seed $SEED --env $ENV --epochs $EPOCHS  --alg $ALG1 --n_runs $N_RUNS --atk_params $C $B1 $B2 --atk_alg $ALG2 --atk_pfm $PFM --dir $DIR  --atk_radius $atk_radius
  fi
fi

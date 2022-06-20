#!/bin/bash

GPU=$1
ENV=$2
SEED=$3
N_RUNS=$4
ALG1="duel"
ALG2="double"
if [ $ENV == "Acrobot-v1" ];then
  B1=400
  B2=4
  EPOCHS=100
  PFM1=-101
  PFM2=-200
  PFM3=-100
  PFM4=-199
  CHANGE=0.9
elif [ $ENV == "LunarLander-v2" ];then
  B1=800
  B2=4
  EPOCHS=150
  PFM1=154
  PFM2=2
  PFM3=202
  PFM4=10
  CHANGE=0.8
elif [ $ENV == "CartPole-v1" ];then
  B1=500
  B2=5
  EPOCHS=100
  PFM1=500
  PFM2=220
  PFM3=500
  PFM4=199
  CHANGE=0.5
elif [ $ENV == "MountainCar-v0" ];then
  B1=100
  B2=2.5
  EPOCHS=100
  PFM1=-108
  PFM2=-158
  PFM3=-101
  PFM4=-156
  CHANGE=0.8
fi

EXP="experiments"
UNDERLINE="_"
EXP_NAME=$ENV$UNDERLINE$SEED
EXP_ALL_FOLDER=../$EXP/
EXP_DIR=$EXP_ALL_FOLDER$EXP_NAME

if [ ! -d $EXP_DIR ];then
  mkdir $EXP_DIR
fi

#cp $(cd `dirname $0`;pwd)/${0##*/} $EXP_DIR/${0##*/}

group_name=$5

if [ $group_name == 0 ];then
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS
  exit 0
fi

C=$6

if [ $# == 7 ];then
  B2=$7
fi

if [ $group_name == 1 ];then
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 uniform
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 promote $ALG2 -2000
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 target $ALG2 -2000
  exit 0
fi
if [ $group_name == 2 ];then
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 offline $ALG2 $PFM1
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 offline $ALG2 $PFM2
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 offline $ALG2 -2000
  exit 0
fi
if [ $group_name == 3 ];then
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 uniform
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 promote $ALG1 -2000
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 target $ALG1 -2000
  exit 0
fi
if [ $group_name == 4 ];then
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 offline $ALG1 $PFM3
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 offline $ALG1 $PFM4
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 offline $ALG1 -2000
  exit 0
fi
if [ $group_name == 5 ];then
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 online $ALG2 $CHANGE
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 online $ALG1 $CHANGE
  exit 0
fi
if [ $group_name == 6 ];then
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 uniform
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 uniform
  exit 0
fi
if [ $group_name == 7 ];then
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 promote $ALG2 -2000
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 target $ALG2 -2000
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 promote $ALG1 -2000
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 target $ALG1 -2000
  exit 0
fi
if [ $group_name == 8 ];then
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 uniform
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 uniform
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 -$B2 uniform
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 -$B2 uniform
  exit 0
fi
if [ $group_name == 9 ];then
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 promote $ALG2 $PFM1
  bash dqn.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 target $ALG2 $PFM1
  exit 0
fi
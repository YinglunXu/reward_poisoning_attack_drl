#!/bin/bash
GPU=$1
ENV=$2
SEED=$3
N_RUNS=$4

if [ $ENV == "HalfCheetah-v2" ];then
  ALG1="ddpg"
  ALG2="td3"
  B1=6300
  B2=42
  EPOCHS=150
  PFM1=6007
  PFM2=12374
  PFM3=5947
  PFM4=12766
  C_list=(0.01 0.005 0.002 0.0005)
elif [ $ENV == "Hopper-v3" ];then
  ALG1="td3"
  ALG2="sac"
  B1=2500
  B2=25
  EPOCHS=150
  PFM1=1828
  PFM2=3619
  PFM3=1801
  PFM4=3562
  C_list=(0.02 0.01 0.005 0.001)
elif [ $ENV == "Walker2d-v2" ];then
  ALG1="td3"
  ALG2="sac"
  B1=2500
  B2=25
  EPOCHS=150
  PFM1=2552
  PFM2=5172
  PFM3=2426
  PFM4=4622
  C_list=(0.01 0.005 0.002 0.0005)
elif [ $ENV == "Swimmer-v3" ];then
  ALG1="ddpg"
  ALG2="ppo"
  B1=80
  B2=0.8
  EPOCHS=150
  PFM1=120
  PFM2=61
  PFM3=120
  PFM4=61
  C_list=(0.015 0.008 0.004 0.001)
fi

EXP="experiments"
UNDERLINE="_"
EXP_NAME=$ENV$UNDERLINE$SEED
EXP_ALL_FOLDER=../$EXP/
EXP_DIR=$EXP_ALL_FOLDER$EXP_NAME

if [ ! -d $EXP_DIR ];then
 mkdir $EXP_DIR
fi

cp $(cd `dirname $0`;pwd)/${0##*/} $EXP_DIR/${0##*/}



group_name=$5
if [ $group_name == 0 ];then
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS
  exit 0
fi

C=$6
if [ $# == 7 ];then
  B2=$7
fi

if [ $group_name == 1 ];then
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 offline $ALG2 $PFM1 $atk_radius
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 offline $ALG2 $PFM2 $atk_radius
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 offline $ALG2 -2000 $atk_radius
  exit 0
fi

if [ $group_name == 2 ];then
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 offline $ALG1 $PFM3 $atk_radius
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 offline $ALG1 $PFM4 $atk_radius
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 offline $ALG1 -2000 $atk_radius
  exit 0
fi

if [ $group_name == 3 ];then
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 uniform
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 promote $ALG2 $atk_radius
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG1 $N_RUNS $C $B1 $B2 target $ALG2 $atk_radius
  exit 0
fi

if [ $group_name == 4 ];then
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 uniform
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 promote $ALG1 $atk_radius
  bash ac.sh $GPU $SEED $ENV $EPOCHS $ALG2 $N_RUNS $C $B1 $B2 target $ALG1 $atk_radius
  exit 0
fi


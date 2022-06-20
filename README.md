# Adversarial MDP attack

Official code base for [Efficient Reward Poisoning Attacks on Online Deep Reinforcement Learning](https://arxiv.org/abs/2205.14842).

# Setup
All python dependencies are in `environment.yml`.

```
conda env create -f environment.yml
```

# Usage

For adversarial attack against environments with discrete actiion space, use `all_dqn.sh`; for environments with continuous action space, use `all_ac.sh`. Taking `all_dqn.sh` as an example, one should run the command in the following format:

```
bash all_dqn.sh $gpu $env $seed $n_runs $group_name $C $B1
```

| Argument              | Description                                                                           |
| ----------------------| --------------------------------------------------------------------------------------|
| $gpu                  | The index of the gpu to run the experiment      |
| $seed    | The random seed                                     |
| $n_runs     | number of times to repeat the experiment under the same setting      |
| $group_name     | the attack methods to experiment with. 0 is no attack; 1 is the UR, RPI, RPP attack; 2 is the LPE attack; 3 is the UR, RPI, RPP attack with another learning algorithm; 4 is the LPE attack with another learning algorithm  |
| $C            | portion of number of steps that can be corrupted at most                     |
|$B1 | the limit on the amount of corruption on each step (optional)|


By default the other parameters for the attack or the learning algorithms are fixed, and one can modify them in the scripts or source code.

The output in the "outputs.txt" file contains the performance of the learned policy by the learning algorithm after each epoch for each run of the experiment. To reproduce the main results in the paper for comparison between the effect of different attack in different environments, one can run the following to commands:

```
python analyze.py --exp $env_seed
```

```
python plot_score.py --exps $env_seed
```

`$env_seed` is the name of environment and the random seed. 

The output plot can be found in the `figure` folder.
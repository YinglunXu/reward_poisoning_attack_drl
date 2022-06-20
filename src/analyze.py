import numpy as np
import argparse
import json
import os


def get_score(file_path):
    with open(file_path,'r') as f:
        data = json.load(f)

    C_log = data['C']
    score_log = data['score']
    pfm = []
    for run_i in score_log:
        run_i = np.array(run_i)
        run_log = []
        for epoch_ret in run_i:
            run_log.append(epoch_ret.mean())
        run_log.sort()
        pfm.append(run_log[-1])
    # print(file_path, ' : ', np.array(pfm).mean())
    return np.array(pfm).mean(), np.array(pfm).std()

# def get_trainlog(file_path):
#     with open(file_path,'r') as f:
#         data = json.load(f)
#
#     C_log = data['C']
#     score_log = data['score']
#     run_logs = []
#     for run_i in score_log:
#         run_i = np.array(run_i)
#         run_log = []
#         for epoch_ret in run_i:
#             run_log.append(epoch_ret.mean())
#         run_logs.append(run_log)
#     n, m = len(run_logs), len(run_logs[0])
#     avg_log = [0 for i in range(m)]
#     for i in range(m):
#         for j in range(n):
#             avg_log[i] += run_logs[j][i]
#         avg_log[i] /= n
#     return avg_log

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default='CartPole-v1_0')
    args = parser.parse_args()

    input_dir = os.path.join('../experiments/', args.exp)


    exps = dict()
    for cur_exp in os.listdir(input_dir):
        if cur_exp.endswith('sh'):
            continue
        if len(cur_exp.split('_')) == 3:
            # CartPole-v1_duel_clean
            group_name = '_'.join(cur_exp.split('_')[1:])
            exps[group_name] = [cur_exp]
        else:
            # CartPole-v1_duel_offline_220_0.02_500_5 --> duel_offline_220
            # CartPole-v1_duel_uniform_0.02_500_5 --> duel_uniform
            group_name = '_'.join(cur_exp.split('_')[1:-3])
            exps.setdefault(group_name,[])
            exps[group_name].append(cur_exp)

    for group_name,group_exp in exps.items():
        if 'clean' not in group_name:
            n = len(exps[group_name])
            break
    for group_name,group_exp in exps.items():
        if 'clean' in group_name:
            exps[group_name] = [exps[group_name][0] for i in range(n)]

    scores = dict()
    errors = dict()
    for group_name,group_exp in exps.items():
        if 'clean' not in group_name:
            exps[group_name] = sorted(group_exp, key=(lambda x:float(x.split('_')[-3])))

        scores[group_name] = [get_score(os.path.join(input_dir, x, 'outputs.json'))[0] for x in exps[group_name]]
        errors[group_name] = [get_score(os.path.join(input_dir, x, 'outputs.json'))[1] for x in exps[group_name]]

    train_logs = dict()
    for group_name,group_exp in exps.items():
        if 'clean' not in group_name:
            exps[group_name] = sorted(group_exp, key=(lambda x:float(x.split('_')[-3])))

        # train_logs[group_name] = [get_trainlog(os.path.join(input_dir, x, 'outputs.json')) for x in exps[group_name]]

    names = dict()
    offline = dict()
    for group_name in exps.keys():
        if 'offline' not in group_name:
            names[group_name] = group_name.split('_')[1]
        else:
            alg, pfm = group_name.split('_')[0], group_name.split('_')[2]
            pfm = int(pfm)
            offline.setdefault(alg,[])
            offline[alg].append(pfm)
            offline[alg].sort()

    for group_name in exps.keys():
        if 'offline' in group_name:
            alg, pfm = group_name.split('_')[0], group_name.split('_')[2]
            pfm = int(pfm)
            for i in range(len(offline[alg])):
                if pfm == offline[alg][i]:
                    if i == 0:
                        names[group_name] = 'random'
                    elif i == 1:
                        names[group_name] = 'high value1'
                    elif i == 2:
                        names[group_name] = 'high value2'

    for group_name in scores:
        print(group_name," : ", scores[group_name]," : ", names[group_name])

    data = dict()
    data['exps'] = exps
    data['scores'] = scores
    data['errors'] = errors
    data['train_logs'] = train_logs
    data['names'] = names

    with open(os.path.join('../figures','%s.json' %(args.exp)), 'w') as f:
        f.write(json.dumps(data))


    # for group_name,group_exp in exps.items():
    #     if 'clean' not in group_name:
    #         x = [float(exp.split('_')[-3]) for exp in group_exp]
    #         break
    # alg1, alg2 = '', ''
    # for _,group_exp in exps.items():
    #     if alg1 == '':
    #         alg1 = group_exp[0].split('_')[1]
    #     elif alg1 != group_exp[0].split('_')[1]:
    #         alg2 = group_exp[0].split('_')[1]
    #         break
    # print(alg1,alg2)
    #
    # f, axs = plt.subplots(2, 1, figsize=(4, 8))
    # for i, ax in enumerate(axs.flat):
    #     ax.set_xlabel('Corrupted Steps C/T')
    #     ax.set_ylabel('Best Learned Reward')
    #     ax.set_xticks(x)
    #     for group_name in scores:
    #         if i == 0 and alg1 in group_name:
    #             ax.plot(x, scores[group_name], label=group_name)
    #         if i == 1 and alg2 in group_name:
    #             ax.plot(x, scores[group_name], label=group_name)
    #     ax.grid(True)
    # f.legend(loc='upper center', ncol=8)
    # plt.savefig('./experiment2.png', dpi=600, bbox_inches='tight')




    # x = [0.02, 0.03, 0.04]
    # colors = ["r.-", "b.--", "k.--", "gv-", "y.--", "m.--", "cv-"]
    # colort = 'y'
    # labels = ["no attack", "uniform", "random", "high-value1", "high-value2", "online", "online2"]
    # titles = ["Acrobot Duel", "LunarLander Duel", "MountainCar Duel", "Pendulum Duel", "CartPole Duel",
    #           "Acrobot Double", "LunarLander Double", "MountainCar Double", "Pendulum_Double", "CartPole_Double"]
    # results = [Acrobot_Duel, LunarLander_Duel, MountainCar_Duel, Pendulum_Duel, CartPole_Duel, Acrobot_Double,
    #            LunarLander_Double, MountainCar_Double, Pendulum_Double, CartPole_Double]
    # f, axs = plt.subplots(2, 5, figsize=(20, 8))
    # for i, ax in enumerate(axs.flat):
    #     ax.set_xlabel('Corrupted Steps C/T')
    #     ax.set_ylabel('$10^{th}$ Best Learned Reward')
    #     ax.set_title(titles[i])
    #     #     ax.set_xticks(np.arange(0.02,0.05,0.01))
    #     ax.set_xticks(x)
    #
    #     for j in range(len(results[i])):
    #         if i == 0:
    #             ax.plot(x, results[i][j], colors[j], label=labels[j])
    #         else:
    #             ax.plot(x, results[i][j], colors[j])
    #     if i == 0:
    #         ax.plot(x, [-180 for i in range(3)], colort, label="threshold")
    #     if i == 5:
    #         ax.plot(x, [-180 for i in range(3)], colort)
    #     if i == 1 or i == 6:
    #         ax.plot(x, [40 for i in range(3)], colort)
    #     if i == 2 or i == 7:
    #         ax.plot(x, [-120 for i in range(3)], colort)
    #     if i == 3 or i == 8:
    #         ax.plot(x, [-280 for i in range(3)], colort)
    #     if i == 4 or i == 9:
    #         ax.plot(x, [400 for i in range(3)], colort)
    #     ax.grid(True)
    #
    # f.legend(loc='upper center', ncol=8)
    # plt.subplots_adjust(left=0.0, bottom=0.15, right=0.93, top=0.9, wspace=0.25, hspace=0.4)
    # plt.savefig('./experiment.png', dpi=600, bbox_inches='tight')
    # plt.show()

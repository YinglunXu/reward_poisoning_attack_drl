import matplotlib.pyplot as plt
import argparse
import json
import os
import time
import matplotlib.colors as colr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exps", nargs="+", type=str)
    parser.add_argument("--name", type=str, default='experiments')
    args = parser.parse_args()

    all_data = []
    for exp in args.exps:\


        path = os.path.join('../figures', exp + '.json')
        with open(path, 'r') as f:
            all_data.append(json.load(f))

    colors = {"clean": colr.cnames['crimson'], "uniform": colr.cnames['olive'], "random": colr.cnames['cornflowerblue'],
              "promote": colr.cnames['teal'], "target": colr.cnames['seagreen'],
              "high value1": colr.cnames['cornflowerblue'], "high value2": colr.cnames['cornflowerblue'], "target2": colr.cnames['lightcoral']}
    labels = {"clean": "None", "uniform": "UR", "random": "LPE(1)", "high value2": "LPE(2)", "high value1": "LPE(3)", "target": "RPI", "promote": "RPP"}
    markers = {"clean": ".", "uniform": "1", "random": "+", "promote": "s", "target": "d",
              "high value1": "v", "high value2": "^", "target2": "^"}
    linestyles = {"clean": "-", "uniform": "-", "random": "-", "promote": "--", "target": "-",
              "high value1": "--", "high value2": "-.", "target2": "--"}

    f, axs = plt.subplots(2, len(all_data), figsize=(4*len(all_data), 8))
    j = 0
    for data in all_data:
        exps = data['exps']
        scores = data['scores']
        names = data['names']
        for group_name, group_exp in exps.items():
            x = [float(exp.split('_')[-3]) for exp in group_exp]
            break
        alg1, alg2 = '', ''
        for _,group_exp in exps.items():
            if alg1 == '':
                alg1 = group_exp[0].split('_')[1]
            elif alg1 != group_exp[0].split('_')[1]:
                alg2 = group_exp[0].split('_')[1]
                break

        # for i, ax in enumerate(axs.flat):
        #     for name in labels:
        #         ax.plot([1,2],[2,3],label=labels[name], c=colors[name], marker=markers[name], linestyle=linestyles[name])
        #     break

        for i, ax in enumerate(axs.flat):
            if i % len(all_data) == j:
                ax.set_xlabel('Corrupted Steps C/T')
                ax.set_ylabel('Best Learned Policy Value')
                if len(x) <= 4:
                    ax.set_xticks(x)
                else:
                    ax.set_xticks([x[0]]+x[3:])
                ax.set_facecolor('xkcd:grey')
                for group_name in scores:
                    if i == j and alg1 in group_name:
                        if j==0:
                            title = args.exps[j].split('_')[0] + '-' + group_name.split('_')[0]
                            ax.set_title(title)
                            # ax.errorbar(x, scores[group_name], yerr=errors[group_name], capsize=4, color=colors[names[group_name]], label=labels[names[group_name]])
                            ax.plot(x, scores[group_name], c=colors[names[group_name]], marker=markers[names[group_name]], linestyle=linestyles[names[group_name]])
                        else:
                            title = args.exps[j].split('_')[0] + '-' + group_name.split('_')[0]
                            ax.set_title(title)
                            # ax.errorbar(x, scores[group_name], yerr=errors[group_name], capsize=4, color=colors[names[group_name]])
                            ax.plot(x, scores[group_name], c=colors[names[group_name]], marker=markers[names[group_name]], linestyle=linestyles[names[group_name]])
                    if i == j + len(all_data) and alg2 in group_name:
                        title = args.exps[j].split('_')[0] + '-' + group_name.split('_')[0]
                        ax.set_title(title)
                        # ax.errorbar(x, scores[group_name], yerr=errors[group_name], capsize=4, color=colors[names[group_name]])
                        ax.plot(x, scores[group_name], c=colors[names[group_name]], marker=markers[names[group_name]], linestyle=linestyles[names[group_name]])
                ax.grid(True)
                ax.set_facecolor(colr.cnames['lightgrey'])
        j += 1
    # f.legend(loc='upper center', ncol=8)
    plt.subplots_adjust(left=0.0, bottom=0.15, right=0.93, top=0.9, wspace=0.25, hspace=0.4)
    plt.savefig('../figures/%s.png' %(args.name), dpi=600, bbox_inches='tight')

    # f, axs = plt.subplots(2, len(all_data), figsize=(4*len(all_data), 8))
    # j = 0
    # for data in all_data:
    #     exps = data['exps']
    #     train_logs = data['train_logs']
    #     alg1, alg2 = '', ''
    #     for _,group_exp in exps.items():
    #         if alg1 == '':
    #             alg1 = group_exp[0].split('_')[1]
    #         elif alg1 != group_exp[0].split('_')[1]:
    #             alg2 = group_exp[0].split('_')[1]
    #             break
    #     for i, ax in enumerate(axs.flat):
    #         if i % len(all_data) == j:
    #             ax.set_xlabel('training step')
    #             ax.set_ylabel('performance')
    #             for group_name in scores:
    #                 if i == j and alg1 in group_name:
    #                     for log in train_logs[group_name]:
    #                         ax.plot([i for i in range(len(log))], log, label=group_name)
    #                 if i == j + len(all_data) and alg2 in group_name:
    #                     for log in train_logs[group_name]:
    #                         ax.plot([i for i in range(len(log))], log, label=group_name)
    #             ax.grid(True)
    #     j += 1
    # # f.legend(loc='upper center', ncol=6)
    # plt.savefig('./experiments_train.png', dpi=600, bbox_inches='tight')
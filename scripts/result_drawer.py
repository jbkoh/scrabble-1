import pdb
import sys
import os
import json
from copy import deepcopy

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 10
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.labelpad'] = 0
plt.rcParams['font.size'] = 10
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.axes import Axes

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, dir_path + '/..')
from scrabble.common import *
from scrabble.eval_func import *


import plotter
from plotter import save_fig

building_anon_map = {
    'ebu3b': 'A-1',
    'uva_cse': 'B-1',
    'sdh': 'C-1'
}
colors = ['firebrick', 'deepskyblue', 'darkgreen', 'goldenrod']
inferencer_names = ['zodiac', 'al_hong', 'scrabble']
LINESTYLES = [':', '--', '-.', '-']
FIG_DIR = './figs'

def average_data(xs, ys, target_x):
    target_y = np.zeros((1, len(target_x)))
    for x, y in zip(xs, ys):
        yinterp = np.interp(target_x, x, y)
        target_y += yinterp / len(ys) * 100
    return target_y.tolist()[0]

def plot_ir2tagsets_only(target_building, source_building):
    linestyles = deepcopy(LINESTYLES)
    outputfile = FIG_DIR + '/ir2tagsets_only.pdf'
    title = ''
    configs = get_ir2tagsets_configs(target_building, source_building)
    xlabel = '# of Samples'
    ylabel = 'Metric (%)'
    fig, ax = plt.subplots(1, 1)
    for config in configs:
        filename = get_filename_for_ir2tagsets(target_building, config)
        with open(filename, 'r') as fp:
            res = json.load(fp)
        accuracy = res['accuracy']
        macrof1 = res['macrof1']
        xticks = [0, 10] + list(range(50, 201, 50))
        xticks_labels = [''] + [str(n) for n in xticks[1:]]
        yticks = range(50,101,10)
        yticks_labels = [str(n) for n in yticks]
        xlim = (50, xticks[-1])
        ylim = ((50, 100))
        #ylim = (yticks[0], yticks[-1])
        interp_x = list(range(10, 200, 5))
        ys = [accuracy, macrof1]
        legends = ''
        _, plots = plotter.plot_multiple_2dline(
            interp_x, ys, xlabel, ylabel, xticks, xticks_labels,
            yticks, yticks_labels, title, ax, fig, ylim, xlim, legends,
            linestyles=[linestyles.pop()]*len(ys), cs=colors)
    ax.grid(True)
    fig.set_size_inches((3,2))
    save_fig(fig, outputfile)

def plot_pointonly_notransfer():
    buildings = ['ebu3b', 'uva_cse', 'sdh', 'ghc']
    #buildings = ['sdh', 'ebu3b']
    outputfile = FIG_DIR + '/pointonly_notransfer.pdf'

    fig, axes = plt.subplots(1, len(buildings))
    xticks = [0, 10] + list(range(50, 251, 50))
    xticks_labels = [''] + [str(n) for n in xticks[1:]]
    yticks = range(0,101,20)
    yticks_labels = [str(n) for n in yticks]
    xlim = (-5, xticks[-1]+5)
    ylim = (yticks[0]-2, yticks[-1]+5)
    interp_x = list(range(10, 250, 5))
    for ax_num, (ax, building) in enumerate(zip(axes, buildings)): # subfigure per building
        xlabel = '# of Samples'
        ylabel = 'Metric (%)'
        title = building_anon_map[building]
        linestyles = deepcopy(LINESTYLES)
        for inferencer_name in inferencer_names:
            if building == 'uva_cse' and inferencer_name == 'scrabble':
                continue
            xs = []
            ys = []
            xss = []
            f1s = []
            mf1s = []
            for i in range(0, EXP_NUM):
                with open('result/pointonly_notransfer_{0}_{1}_{2}.json'
                          .format(inferencer_name, building, i)) as  fp:
                    data = json.load(fp)
                xss.append([datum['learning_srcids'] for datum in data])
                if inferencer_name == 'al_hong':
                    f1s.append([datum['metrics']['f1_micro'] for datum in data])
                    mf1s.append([datum['metrics']['f1_macro'] for datum in data])
                else:
                    f1s.append([datum['metrics']['f1'] for datum in data])
                    mf1s.append([datum['metrics']['macrof1'] for datum in data])
            xs = xss[0] # Assuming all xss are same.
            f1 = average_data(xss, f1s, interp_x)
            mf1 = average_data(xss, mf1s, interp_x)
            x = interp_x
            ys = [f1, mf1]
            if ax_num == 0:
                #data_labels = ['Baseline Acc w/o $B_s$',
                #               'Baseline M-$F_1$ w/o $B_s$']
                legends = ['MicroF1, {0}'.format(inferencer_name),
                           'MacroF1, {0}'.format(inferencer_name)
                           ]
            else:
                #data_labels = None
                legends = None

            _, plots = plotter.plot_multiple_2dline(
                x, ys, xlabel, ylabel, xticks, xticks_labels,
                yticks, yticks_labels, title, ax, fig, ylim, xlim, legends,
                linestyles=[linestyles.pop()]*len(ys), cs=colors)
    for ax in axes:
        ax.grid(True)
    for i in range(1,len(buildings)):
        axes[i].set_yticklabels([])
        axes[i].set_ylabel('')
    for i in range(0,len(buildings)):
        if i != 1:
            axes[i].set_xlabel('')
    axes[0].legend(bbox_to_anchor=(3.2, 1.5), ncol=3, frameon=False)
    fig.set_size_inches((8,2))
    save_fig(fig, outputfile)

def plot_pointonly_transfer():
    buildings = ['ebu3b', 'uva_cse']
    outputfile = FIG_DIR + '/pointonly_transfer.pdf'

    fig, axes = plt.subplots(1, len(buildings))
    xticks = [0, 10] + list(range(50, 251, 50))
    xticks_labels = [''] + [str(n) for n in xticks[1:]]
    yticks = range(0,101,20)
    yticks_labels = [str(n) for n in yticks]
    xlim = (-5, xticks[-1]+5)
    ylim = (yticks[0]-2, yticks[-1]+5)
    interp_x = list(range(10, 250, 5))
    for i, (ax, building) in enumerate(zip(axes, buildings)): # subfigure per building
        xlabel = '# of Samples'
        ylabel = 'Metric (%)'
        title = building_anon_map[building]
        linestyles = deepcopy(LINESTYLES)
        for inferencer_name in inferencer_names:
            xs = []
            ys = []
            xss = []
            f1s = []
            mf1s = []
            for i in range(0, EXP_NUM):
                with open('result/pointonly_transfer_{0}_{1}_{2}.json'
                          .format(inferencer_name, building, i)) as  fp:
                    data = json.load(fp)
                xss.append([datum['learning_srcids'] for datum in data])
                if inferencer_name == 'al_hong':
                    f1s.append([datum['metrics']['f1_micro'] for datum in data])
                    mf1s.append([datum['metrics']['f1_macro'] for datum in data])
                else:
                    f1s.append([datum['metrics']['f1'] for datum in data])
                    mf1s.append([datum['metrics']['macrof1'] for datum in data])
            xs = xss[0] # Assuming all xss are same.
            f1 = average_data(xss, f1s, interp_x)
            mf1 = average_data(xss, mf1s, interp_x)
            x = interp_x
            ys = [f1, mf1]
            if i == 2:
                #data_labels = ['Baseline Acc w/o $B_s$',
                #               'Baseline M-$F_1$ w/o $B_s$']
                legends = ['MicroF1, {0}'.format(inferencer_name),
                           'MacroF1, {0}'.format(inferencer_name)
                           ]
            else:
                #data_labels = None
                legends = None

            _, plots = plotter.plot_multiple_2dline(
                x, ys, xlabel, ylabel, xticks, xticks_labels,
                yticks, yticks_labels, title, ax, fig, ylim, xlim, legends,
                linestyles=[linestyles.pop()]*len(ys), cs=colors)
    save_fig(fig, outputfile)

def get_ir2tagsets_configs(target_building, source_building):
    configs = [
            {'use_brick_flag': True,
             'negative_flag': True,
             'source_building_list': [source_building, target_building],
             'target_building': target_building,
             'tagset_classifier_type': 'MLP',
             'task': 'ir2tagsets'
             },
            {'use_brick_flag': True,
             'negative_flag': True,
             'source_building_list': [target_building],
             'target_building': target_building,
             'tagset_classifier_type': 'MLP',
             'task': 'ir2tagsets'
             },
            #{'use_brick_flag': True,
            # 'negative_flag': True,
            # 'source_building_list': [target_building],
            # 'target_building': target_building,
            # 'tagset_classifier_type': 'StructuredCC',
            # 'task': 'ir2tagsets'
            # },
            {'use_brick_flag': False,
             'negative_flag': False,
             'source_building_list': [target_building],
             'target_building': target_building,
             'tagset_classifier_type': 'MLP',
             'task': 'ir2tagsets'
             },
            ]
    return configs

def get_filename_for_ir2tagsets(target_building, config):
    filename = 'result/ir2tagsets_{target}_{source}_{sample_aug}_{ct}'\
            .format(
                    target = target_building,
                    source = config['source_building_list'][0],
                    sample_aug = 'sampleaug' if config['use_brick_flag'] 
                                 else 'noaug',
                    ct = config['tagset_classifier_type'].lower()
                    )
    return filename

def calculate_ir2tagets_results(target_building,
                                source_building,
                                recalculate=False):
    EXP_NUM = 2
    default_configs = {
            'use_known_tags': True,
            'task': 'ir2tagsets'
            }
    configs = get_ir2tagsets_configs(target_building, source_building)
    for new_config in configs:
        filename = get_filename_for_ir2tagsets(target_building, new_config)
        #if os.path.isfile(filename):
        #    print('{0} exists.'.format(filename))
        fig, ax = plt.subplots(1, 1)
        interp_x = list(range(10, 200, 5))
        xss = []
        accs = []
        mf1s = []
        config = deepcopy(default_configs)
        config.update(new_config)
        for i in range(0, EXP_NUM):
            config['postfix'] = str(i)
            res = query_result(config)
            history = res['history']
            if not history:
                pdb.set_trace()
            x = []
            acc = []
            mf1 = []
            for hist in history:
                pred = hist['pred']
                truth = get_true_labels(pred.keys(), 'tagsets')
                acc.append(get_accuracy(truth, pred))
                mf1.append(get_macro_f1(truth, pred))
                num_learning_srcids = hist['learning_srcids']
                if len(config['source_building_list']) == 2:
                    num_learning_srcids -= 200
                elif len(config['source_building_list']) > 2:
                    raise Exception('define this case')
                x.append(num_learning_srcids)
            mf1s.append(mf1)
            accs.append(acc)
            xss.append(x)
        averaged_acc = average_data(xss, accs, interp_x)
        averaged_mf1 = average_data(xss, mf1s, interp_x)
        res = {
                'accuracy': averaged_acc,
                'macrof1': averaged_mf1
                }
        with open(filename, 'w') as fp:
            json.dump(res, fp)

if __name__ == '__main__':
    calculate_ir2tagets_results('ebu3b', 'ap_m')
    #plot_ir2tagsets_only('ebu3b', 'ap_m')
    #plot_pointonly_notransfer()
    #plot_pointonly_transfer()

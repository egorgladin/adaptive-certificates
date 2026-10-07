"""Export the two-panel figure, endpoint CSV and the three tables."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np



def export(root):
    root = Path(root).resolve()
    rows = [json.loads(p.read_text()) for p in sorted(root.glob('d*/result.json'))]
    index = {(r['d'],r['k'],r['arm']):r for r in rows}
    endpoints = []
    for row in rows:
        for eps in row['targets']:
            hit = row['hits'].get(str(eps))
            endpoints.append({'d':row['d'], 'k':row['k'], 'arm':row['arm'], 'epsilon':eps,
                'certified':hit is not None, 'calls':None if hit is None else hit['calls'],
                'certificate':None if hit is None else hit['certificate'],
                'actual_gap':None if hit is None else hit['actual_gap'],
                'm':None if hit is None else hit['m'], 'q':None if hit is None else hit['q'],
                'seconds':None if hit is None else hit['seconds'],
                'retained_bytes':None if hit is None else hit['retained_bytes']})
    with (root/'endpoints.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(endpoints[0])); writer.writeheader()
        writer.writerows(endpoints)
    def calls(d,k,arm,eps=1e-4):
        return index[d,k,arm]['hits'][str(eps)]['calls']
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':11})
    fig, axes = plt.subplots(1,2,figsize=(10.3,3.7),layout='constrained')
    ranks = [2,4,8,16]
    for d,color,marker,style in [(64,'#0072B2','o','-'),(256,'#D55E00','s','--'),
                                 (1024,'#009E73','^',':')]:
        axes[0].plot(ranks,[calls(d,k,'adaptive') for k in ranks],style,
                     color=color,marker=marker,markersize=7,markerfacecolor='none',
                     linewidth=1.7,label=f'Adaptive, d={d}')
    axes[0].plot(ranks,[calls(256,k,'ideal') for k in ranks],'--',color='#555555',
                 marker='x',linewidth=1.6,label='Known subspace')
    axes[0].set(xlabel='Hidden dimension k',ylabel='Objective-oracle calls',xticks=ranks,
                title=r'(a) Dimension sweep, $\varepsilon=10^{-4}$')
    axes[0].text(.04,.92,'Discovered m = k in every run',transform=axes[0].transAxes,fontsize=9)
    axes[0].legend(loc='lower right',fontsize=8.3,frameon=False)
    eps = [1e-2,1e-3,1e-4,1e-5,1e-6]
    for arm,label,color,marker,style in [('adaptive','Adaptive','#0072B2','o','-'),
                                        ('ideal','Known subspace','#555555','x','--')]:
        axes[1].plot(range(2,7),[calls(256,8,arm,e) for e in eps],style,
                     color=color,marker=marker,linewidth=1.7,label=label)
    axes[1].set(xlabel=r'$\log_{10}(1/\varepsilon)$',ylabel='Objective-oracle calls',
                xticks=range(2,7),title='(b) Accuracy sweep, d=256, k=8')
    axes[1].legend(loc='upper left',frameon=False,fontsize=9)
    for ax in axes:
        ax.grid(axis='y',alpha=.22)
    fig.savefig(root/'adaptive_cp.pdf')
    plt.close(fig)
    dimension_table = '| d | k=2 | k=4 | k=8 | k=16 |\n|---:|---:|---:|---:|---:|\n'
    for d in [64,256,1024]:
        dimension_table += f'| {d} | '+' | '.join(str(calls(d,k,'adaptive')) for k in ranks)+' |\n'
    dimension_table += '| Known subspace | '+' | '.join(str(calls(256,k,'ideal')) for k in ranks)+' |\n'
    ambient_table = '| d | Known subspace | Adaptive | Ambient | Ambient / Adaptive |\n|---:|---:|---:|---:|---:|\n'
    for d in [16,32,64]:
        ambient_table += f'| {d} | {calls(d,4,"ideal")} | {calls(d,4,"adaptive")} | {calls(d,4,"ambient")} | {calls(d,4,"ambient")/calls(d,4,"adaptive"):.2f} |\n'
    accuracy_table = '| ε | Known subspace | Adaptive |\n|---:|---:|---:|\n'
    for e in eps:
        accuracy_table += f'| {e:g} | {calls(256,8,"ideal",e)} | {calls(256,8,"adaptive",e)} |\n'
    for name, table in [('dimension_table.md',dimension_table),('ambient_table.md',ambient_table),
                        ('accuracy_table.md',accuracy_table)]:
        (root/name).write_text(table,encoding='utf-8')
    print(f'{len(rows)} runs; {sum(e["certified"] for e in endpoints)}/{len(endpoints)} certified endpoints.')
    print('Tables and figure:',root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--results',required=True)
    export(parser.parse_args().results)

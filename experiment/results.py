"""Tables and figures generated from a fresh experiment run."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('output')
RESULTS = ROOT/'results'
DATASETS = ('abalone','airfoil','adult','breast_cancer')
METHODS = ('subgradient','momentum','adagrad')
NAMES = dict(abalone='Abalone',airfoil='Airfoil',adult='Adult',breast_cancer='Breast Cancer',
             full='Full',compressed='Compressed',single_cut='Single-cut',
             subgradient='SG',momentum='Momentum',adagrad='AdaGrad')
COLORS = dict(full='#2266aa',compressed='#229966',single_cut='#bb5522',
              subgradient='#2266aa',momentum='#dd7722',adagrad='#229966')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def hit(cell,eps=.001):
    return cell['first_hits'].get(str(float(eps)))


def median(values):
    return float(np.median(values)) if values else None


def projected(h):
    return h['seconds']+h['bytes']/2**20 if h else None


def write_csv(name,rows):
    with (ROOT/name).open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(models):
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':10,
                         'legend.fontsize':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.65),sharey=True)
    for ax,dataset in zip(axes,('abalone','adult')):
        for arm in ('full','compressed','single_cut'):
            path=RESULTS/'sensitivity'/f'{dataset}_subgradient_{arm}'
            cell=read(path/'result.json')
            trace=np.load(path/'trace.npy')
            U=np.minimum.accumulate(trace[:,0])
            lower=np.maximum.accumulate(np.maximum(0,trace[:,0]-trace[:,2]/(2*cell['config']['model_lambda'])))
            for check in cell['checks']:
                index=check['records']-1
                lower[index:]=np.maximum(lower[index:],check['lower'])
            ax.semilogy(np.arange(1,len(U)+1),(U-lower)/cell['initial'],
                        color=COLORS[arm],label=NAMES[arm],linewidth=1.25)
        ax.axhline(.001,color='#555555',linestyle='--',linewidth=.8)
        ax.axhline(.0001,color='#777777',linestyle=':',linewidth=.8)
        ax.set_title(NAMES[dataset]+' / SG')
        ax.set_xlabel('Function--subgradient queries')
        ax.set_xlim(0,2000)
        ax.grid(alpha=.18)
        ax.legend(loc='upper right')
    axes[0].set_ylabel(r'Certificate gap / $F(0)$')
    fig.tight_layout(pad=.5)
    fig.savefig(ROOT/'gaps.pdf',bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.),sharex=True)
    for ax,dataset in zip(axes.flat,DATASETS):
        for item in (m for m in models if m['dataset']==dataset):
            if item['values'] is None:
                continue
            ax.plot(range(4),item['values'],'-o',color=COLORS[item['method']],
                    label=NAMES[item['method']],markersize=3.5,linewidth=1.2)
            if item['measured'] is not None:
                ax.scatter([1],[item['measured']],marker='D',s=28,facecolors='white',
                           edgecolors=COLORS[item['method']],zorder=5)
        ax.axhline(1,color='#555555',linestyle='--',linewidth=.8)
        ax.set_title(NAMES[dataset])
        ax.set_xticks(range(4),['0.1','1','10','No delay'])
        ax.set_xlabel('Certificate bandwidth (MiB/s)')
        ax.set_ylabel(r'$T_{\mathrm{comp}}/T_{\mathrm{full}}$')
        ax.grid(alpha=.18)
        ax.legend(loc='upper left')
    fig.tight_layout(pad=.6)
    fig.savefig(ROOT/'bandwidth.pdf',bbox_inches='tight')
    plt.close(fig)


def main():
    rates,taus=read(RESULTS/'rates.json'),read(RESULTS/'thresholds.json')
    rows,settings,models,sensitivity=[],[],[],[]
    for dataset in DATASETS:
        meta=read(ROOT/'data'/dataset/'metadata.json')
        for method in METHODS:
            key=f'{dataset}_{method}'
            settings.append(dict(dataset=dataset,method=method,train=meta['train'],validation=meta['validation'],
                                 test=meta['test'],sigma=meta['sigma'],model_lambda=meta['lam'],
                                 step_scale=rates[key]['lr'],tau_multiplier=taus[key]['multiplier'],tau=taus[key]['tau']))
            cells={a:[read(RESULTS/'confirmation'/f'{key}_repeat{r}_{a}'/'result.json') for r in range(3)]
                   for a in ('full','compressed','single_cut')}
            row=dict(dataset=dataset,method=method,target=cells['full'][0]['targets'][0])
            for arm,entries in cells.items():
                hits=[hit(c) for c in entries if hit(c)]
                times=[h['seconds'] for h in hits]
                row.update({arm+'_successes':len(hits),arm+'_seconds':median(times),
                            arm+'_min_seconds':min(times) if times else None,arm+'_max_seconds':max(times) if times else None,
                            arm+'_queries':entries[0]['records'],arm+'_gap':median([c['gap'] for c in entries]),
                            arm+'_U':entries[0]['U'],arm+'_bytes':median([c['bytes'] for c in entries]),
                            arm+'_test_metric':median([c['test_metric'] for c in entries])})
            comparable=row['full_successes']==row['compressed_successes']==3
            measured=row['compressed_seconds']/row['full_seconds'] if comparable else None
            row['time_saving']=1-measured if comparable else None
            row['byte_ratio']=row['full_bytes']/row['compressed_bytes'] if comparable else None
            row['k']=cells['compressed'][0]['k']
            rows.append(row)
            full=hit(read(RESULTS/'pilot'/f'{key}_full'/'result.json'))
            comp=hit(read(Path(taus[key]['path'])/'result.json'))
            values=[(comp['seconds']+comp['bytes']/(b*2**20))/(full['seconds']+full['bytes']/(b*2**20))
                    for b in (.1,1.,10.)]+[comp['seconds']/full['seconds']] if full and comp else None
            models.append(dict(dataset=dataset,method=method,values=values,measured=measured))
        if dataset in ('abalone','adult'):
            full=hit(read(RESULTS/'pilot'/f'{dataset}_subgradient_full'/'result.json'))
            single=hit(read(RESULTS/'pilot'/f'{dataset}_subgradient_single_cut'/'result.json'))
            for option in taus[f'{dataset}_subgradient']['candidates']:
                comp=hit(read(Path(option['path'])/'result.json'))
                sensitivity.append(sensitivity_row(dataset,'tau_multiplier',option['multiplier'],full,comp,single))
            cells=[read(RESULTS/'sensitivity'/f'{dataset}_subgradient_{a}'/'result.json')
                   for a in ('full','compressed','single_cut')]
            for eps in (.01,.001,.0001):
                sensitivity.append(sensitivity_row(dataset,'relative_accuracy',eps,*[hit(c,eps) for c in cells]))
    write_csv('main.csv',rows)
    write_csv('settings.csv',settings)
    write_csv('sensitivity.csv',sensitivity)
    figures(models)
    successes=[r for r in rows if r['time_saving'] is not None]
    print(f"Compressed faster in {sum(r['time_saving']>0 for r in successes)}/{len(successes)} comparisons at 1 MiB/s.")
    print('Tables and figures:',ROOT.resolve())


def sensitivity_row(dataset,parameter,value,full,comp,single):
    return dict(dataset=dataset,parameter=parameter,value=value,
                full_queries=full['records'] if full else None,compressed_queries=comp['records'] if comp else None,
                single_cut_queries=single['records'] if single else None,k=comp['k'] if comp else None,
                projected_ratio=projected(comp)/projected(full) if full and comp else None,
                single_cut_projected_seconds=projected(single))


if __name__=='__main__':
    main()

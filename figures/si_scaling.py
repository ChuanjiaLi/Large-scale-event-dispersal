"""SI Figures S6, S11 and S12, Table S5 and response-letter Table R2 (everyday-share allocation).

Inputs: results/allocation/{event_metrics, similarity_summary}.csv (src/s01) and results/scaling/*.csv (src/s02).
Outputs: figures/output/si/
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

ROOT=Path(__file__).resolve().parents[1]
MODEL='proportional'
r=ROOT/'results';out=ROOT/'figures/output/si';out.mkdir(parents=True,exist_ok=True)
e=pd.read_csv(r/'allocation/event_metrics.csv').rename(columns={'q':'total_ratio','venue_id':'venue'})
s=pd.read_csv(r/'allocation/similarity_summary.csv').query("subset == 'all'").rename(columns={'ci025':'cluster_ci025','ci975':'cluster_ci975'})
d=pd.read_csv(r/'scaling/event_metrics.csv').query("model == @MODEL and target == 'total'")
reg=pd.read_csv(r/'scaling/scale_error_regressions.csv')
colors=['#5B9FE5','#EF7878','#EEB34D','#59B493'];ink='#252525'
plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.labelsize':8,
 'axes.titlesize':8,'axes.linewidth':.65,'xtick.labelsize':7,'ytick.labelsize':7,
 'xtick.major.width':.65,'ytick.major.width':.65,'xtick.direction':'out',
 'ytick.direction':'out','xtick.major.size':3,'ytick.major.size':3,
 'text.color':ink,'axes.labelcolor':ink,'axes.edgecolor':ink,'svg.fonttype':'none',
 'savefig.facecolor':'white','legend.fontsize':7,'legend.frameon':False})

def letter(ax,x):ax.text(-.19,1.07,x,transform=ax.transAxes,weight='bold',fontsize=10,va='top')
def save(fig,name):
 fig.savefig(out/(name+'.png'),dpi=450,bbox_inches='tight',pad_inches=.04)
 fig.savefig(out/(name+'.svg'),bbox_inches='tight',pad_inches=.04)
 plt.close(fig)

# S11: equal cell domains, paired records and a volume-ratio check.
fig,axs=plt.subplots(1,2,figsize=(6.85,2.65),gridspec_kw={'wspace':.43})
fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.88)
ax=axs[0];rng=np.random.default_rng(20261117);jitter=rng.uniform(-.15,.15,len(e))
for offset,row in zip(jitter,e.itertuples()):ax.plot([offset,1+offset],[row.total_pearson,row.change_pearson],color='#d8d8d8',lw=.25,zorder=0,alpha=.22)
for j,(col,color) in enumerate([('total_pearson',colors[0]),('change_pearson',colors[1])]):
 ax.scatter(j+jitter,e[col],s=4,color=color,alpha=.45,linewidths=0)
 v=s.query('metric == @col').iloc[0]
 ax.errorbar(j,float(v['median']),yerr=[[v['median']-v.cluster_ci025],[v.cluster_ci975-v['median']]],fmt='o',color=ink,markersize=4,capsize=4,lw=1.2,zorder=4)
 ax.text(j,1.075,f"Median {v['median']:.3f}",ha='center',fontsize=7)
ax.set(xticks=[0,1],xticklabels=['Total flow','Extra trips'],xlim=(-.4,1.4),ylim=(-1.03,1.17),ylabel='Pearson correlation with baseline')
letter(ax,'a')
ax=axs[1];mask=e.total_ratio>3
ax.scatter(e.loc[~mask,'total_ratio'],e.loc[~mask,'change_pearson'],s=8,c=colors[0],alpha=.5,linewidths=0)
ax.scatter(e.loc[mask,'total_ratio'],e.loc[mask,'change_pearson'],s=10,c=colors[1],alpha=.7,linewidths=0)
ax.axvline(3,color='#666666',ls='--',lw=.8)
ax.set(xscale='log',ylim=(-1.03,1.07),xlabel=r'Relative surge, $q=M_e/M_b$',ylabel='Extra-trip correlation with baseline')
ax.text(.98,.06,'q > 3: 66 records\nMedian correlation = 0.737',transform=ax.transAxes,ha='right',fontsize=7)
letter(ax,'b');save(fig,'Figure_S11_background')

# S12: all 433 points retained, identical baseline scale; show absolute errors.
fig,axs=plt.subplots(2,3,figsize=(7.1,4.8));fig.subplots_adjust(left=.09,right=.98,bottom=.12,top=.94,wspace=.46,hspace=.51)
xx=np.geomspace(d.baseline_total.min(),d.baseline_total.max(),200)
labels={'fvu':'FVU of everyday-share allocation','mae':'MAE (grid-flow units)','rmse':'RMSE (grid-flow units)','sse':'SSE (squared grid-flow units)','sst':'SST (squared grid-flow units)'}
for ax,key,lab in zip(axs.flat,list(labels),'abcde'):
 ax.scatter(d.baseline_total,d[key],s=5,color=colors[0],alpha=.45,linewidths=0)
 b,a=np.polyfit(np.log(d.baseline_total),np.log(d[key]),1)
 ax.plot(xx,np.exp(a)*xx**b,color=ink,lw=1)
 rr=reg[(reg.model==MODEL)&(reg.subset=='all433')&(reg.specification=='baseline_only')&(reg.dependent=='log_'+key)]
 rr=rr.iloc[0]
 if key=='fvu':rr=pd.read_csv(r/'scaling/power_slope_bootstrap.csv').query("model == @MODEL").iloc[0]
 ax.text(.03,.04,f"b = {b:.3f}\n95% CI [{rr.ci025:.3f}, {rr.ci975:.3f}]",transform=ax.transAxes,fontsize=6.5)
 ax.set(xscale='log',yscale='log',xlabel=r'Baseline grid total, $M_b$',ylabel=labels[key]);letter(ax,lab)
 ax.xaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.xaxis.set_minor_formatter(NullFormatter())
ax=axs[1,2];bins=[(1,1.5),(1.5,2),(2,3),(3,np.inf)]
for (lo,hi),c in zip(bins,colors):
 z=d[(d.total_ratio>lo)&(d.total_ratio<=hi)];b,a=np.polyfit(np.log(z.baseline_total),np.log(z.fvu),1)
 xr=np.geomspace(z.baseline_total.min(),z.baseline_total.max(),100)
 ax.scatter(z.baseline_total,z.fvu,s=4,c=c,alpha=.25,linewidths=0)
 label=(f'{lo:g} < q ≤ {hi:g}' if np.isfinite(hi) else 'q > 3')+f' (n = {len(z)})'
 ax.plot(xr,np.exp(a)*xr**b,lw=1.2,c=c,label=label)
z=d[d.total_ratio<=1];ax.scatter(z.baseline_total,z.fvu,s=12,c='#777777',marker='x',lw=.65,label='q ≤ 1 (n = 3)')
ax.legend(loc='lower left',fontsize=6,handlelength=1.4,borderaxespad=.3)
ax.set(xscale='log',yscale='log',xlabel=r'Baseline grid total, $M_b$',ylabel='FVU by relative surge');letter(ax,'f')
save(fig,'Figure_S12_error_decomposition')

# S6: (a) power-law fit in each city (style of Figure 3c); (b) slopes with uncertainty and sample sizes.
from scipy import stats
c=pd.read_csv(r/'scaling/city_slopes.csv').query("model == @MODEL")
names={'Haerbin':'Harbin','Xian':"Xi'an"}
cities = c.assign(name=c.city.replace(names)).loc[c.eligible]
alphabetical, cities = cities.sort_values('name'), cities.sort_values('beta')
plt.rcParams['mathtext.fontset'] = 'custom'; plt.rcParams['mathtext.rm'] = 'Arial'; plt.rcParams['mathtext.it'] = 'Arial:italic'
fig = plt.figure(figsize=(7.2, 8.9))
top = fig.add_gridspec(3, 5, left=.075, right=.99, top=.965, bottom=.485, hspace=.50, wspace=.40)
for i, row in enumerate(alphabetical.itertuples()):
    ax = fig.add_subplot(top[i // 5, i % 5])
    z = d[d.city == row.city]
    fit = stats.linregress(np.log10(z.baseline_total), np.log10(z.fvu))
    ax.scatter(z.baseline_total, z.fvu, color='#555555', s=5, alpha=.7, linewidths=0, zorder=1)
    xr = np.geomspace(z.baseline_total.min(), z.baseline_total.max() * 1.1, 100)
    ax.plot(xr, 10 ** fit.intercept * xr ** fit.slope, color='#C91E23', ls='--', lw=1.1, zorder=2)
    ax.set(xscale='log', yscale='log')
    lo, hi = np.log10(z.fvu.min()), np.log10(z.fvu.max())
    ax.set_ylim(10 ** (lo - .12 * (hi - lo)), 10 ** (hi + .50 * (hi - lo)))       # room for the annotation
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6)); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=5)); ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(direction='in', which='both', width=.5, labelsize=7)
    ax.text(.96, .95, f'$R^2 = {fit.rvalue ** 2:.3f}$\n' + ('$p < 0.001$' if fit.pvalue < .001 else f'$p = {fit.pvalue:.3f}$'),
            transform=ax.transAxes, ha='right', va='top', fontsize=7, linespacing=1.25)
    ax.set_title(row.name, fontsize=8, pad=3)
    if i % 5 == 0: ax.set_ylabel(r'FVU (1 - $R^2$)', fontsize=8)
    if i == 0: first = ax
fig.text(.53, .448, 'Baseline traffic magnitude (trips)', ha='center', fontsize=8)
first.text(-.42, 1.17, 'a', transform=first.transAxes, weight='bold', fontsize=10, va='top')

y = np.arange(len(cities))
ax = fig.add_axes([.235, .050, .52, .355])
ax.vlines(0, -.5, len(cities) - .45, colors='#858585', linestyles='--', linewidth=.8, zorder=0)
ax.errorbar(cities.beta, y, xerr=[cities.beta - cities.ci025, cities.ci975 - cities.beta],
            fmt='o', markersize=4.2, color='#367BB5', ecolor='#525A60',
            elinewidth=.95, capsize=2, capthick=.85)
ax.set_xlim(cities.ci025.min() - .16, max(0, cities.ci975.max()) + .16)
ax.set_ylim(len(cities) - .45, -1.55)
ax.set_xlabel('Slope of log FVU versus log baseline volume', fontsize=8)
ax.set_yticks([])
ax.spines[['left','right','top']].set_visible(False)
ax.tick_params(axis='x', labelsize=7)
row_coords = ax.get_yaxis_transform()
city_x, records_x, venues_x = -.31, 1.11, 1.30
for x, label, align in [(city_x, 'City', 'left'), (records_x, 'Records', 'center'), (venues_x, 'Venues', 'center')]:
    ax.text(x, -1.12, label, transform=row_coords, ha=align, va='center', fontsize=8, fontweight='bold')
ax.plot([city_x, venues_x + .07], [-.60, -.60], transform=row_coords, color='#BEC4C8', linewidth=.6, clip_on=False)
for i, row in enumerate(cities.itertuples()):
    ax.text(city_x, i, row.name, transform=row_coords, ha='left', va='center', fontsize=8)
    ax.text(records_x, i, str(row.n_events), transform=row_coords, ha='center', va='center', fontsize=8)
    ax.text(venues_x, i, str(row.n_venues), transform=row_coords, ha='center', va='center', fontsize=8)
fig.text(first.get_position().x0 - .42 * first.get_position().width, .418, 'b', weight='bold', fontsize=10, va='top')
save(fig, 'Figure_S6_city_fits')

# Editable table S5: slopes, curve comparison and scale adjustments.
bs=pd.read_csv(r/'scaling/power_slope_bootstrap.csv').set_index('model')
cv=pd.read_csv(r/'scaling/curve_comparison.csv').pivot(index='model',columns='curve',values='venue_cv_log_rmse')
modnames={'proportional':'Everyday-share allocation','xgb':'XGBoost'}
lines=[r'\begin{addedblock}',r'\noindent\textbf{Table S5. Scaling of FVU with baseline volume: everyday-share allocation and XGBoost.}',r'\par\smallskip',r'{\small',r'\textbf{A. Power-law slopes and validation RMSE for $\log FVU$ at held-out venues.}\par',r'\begin{tabular}{@{}lcccc@{}}\toprule',r'Predictor & Slope $b$ [95\% CI] & Power & Exponential & Power + floor \\ \midrule']
for m,name in modnames.items():
 v=bs.loc[m];z=cv.loc[m];lines.append(f'{name} & {v.beta:.3f} [{v.ci025:.3f}, {v.ci975:.3f}] & {z["power"]:.3f} & {z["exponential"]:.3f} & {z["power_with_floor"]:.3f} '.replace('-0.',r'\(-\)0.').replace('-1.',r'\(-\)1.')+r'\\')
lines += [r'\bottomrule\end{tabular}\par\medskip',r'\textbf{B. Adjusted baseline-volume coefficients [95\% CI].}\par',r'\begin{tabular}{@{}lcc@{}}\toprule',r'Predictor & Adjusted for $\log q$ and city & Adjusted for $\log M_e$ and city \\ \midrule']
for m,name in modnames.items():
 vals=[]
 for spec in ['relative_impact_city','event_size_city']:
  z=reg[(reg.model==m)&(reg.subset=='all433')&(reg.specification==spec)&(reg.dependent=='log_fvu')&(reg.term=='log_baseline')].iloc[0]
  vals.append(f'{z.coefficient:.3f} [{z.ci025:.3f}, {z.ci975:.3f}]')
 lines.append(name+' & '+' & '.join(vals).replace('-',r'\(-\)')+r' \\')
lines += [r'\bottomrule\end{tabular}\par}',r'\end{addedblock}']
(out/'Table_S5_scaling.tex').write_text('\n'.join(lines)+'\n')
lines=[r'\par\Needspace{9\baselineskip}\smallskip',r'\textbf{Table R2. Error diagnostics of the everyday-share allocation across 433 records.}\par',r'{\small\begin{tabular}{@{}lcc@{}}\toprule',r'Measure & Log--log slope versus $M_b$ & 95\% interval \\ \midrule']
for key,label in [('fvu','FVU'),('mae','MAE'),('rmse','RMSE'),('sse','SSE'),('sst','SST')]:
 z=reg[(reg.model==MODEL)&(reg.subset=='all433')&(reg.specification=='baseline_only')&(reg.dependent=='log_'+key)].iloc[0]
 coef=z.coefficient
 if key=='fvu':z=bs.loc[MODEL];coef=z.beta
 lines.append(f'{label} & {coef:.3f} & [{z.ci025:.3f}, {z.ci975:.3f}] '+r'\\')
lines += [r'\bottomrule\end{tabular}\par}',r'\smallskip']
(out/'Table_R2_error_summary.tex').write_text('\n'.join(lines)+'\n')
print('S6, S11, S12, Table S5 and Table R2 written to', out)

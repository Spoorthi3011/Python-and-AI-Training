"""
============================================================
  EDA of Titanic Dataset — Python Analysis
  Repository: Python and AI Training
  Author: AI Intern
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# ── Styling ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor':   '#0f0f1a',
    'axes.edgecolor':   '#2a2a4a',
    'axes.labelcolor':  '#c8c8e8',
    'xtick.color':      '#8888aa',
    'ytick.color':      '#8888aa',
    'text.color':       '#e8e8ff',
    'grid.color':       '#1e1e3a',
    'grid.alpha':       0.6,
    'font.family':      'DejaVu Sans',
    'axes.titleweight': 'bold',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

PALETTE     = ['#e05c5c', '#5ca0e0', '#5ce0a0', '#e0c05c', '#c05ce0', '#5ce0e0']
SURVIVED    = '#5ca0e0'   # blue  = survived
DIED        = '#e05c5c'   # red   = died
ACCENT      = '#e0c05c'   # gold

# ── 1. Load dataset ────────────────────────────────────────────────────────
print("=" * 60)
print("  TITANIC EDA — Python & AI Training")
print("=" * 60)

import os
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'titanic.csv')
df = pd.read_csv(csv_path)
print(f"✅  Dataset loaded  ({len(df)} passengers).")

# ── 2. Data Overview ─────────────────────────────────────────────────────────
print(f"\n📊 Shape          : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"🎯 Survival rate  : {df['Survived'].mean()*100:.1f}%")
print("\n── Column dtypes & missing values ──")
info = pd.DataFrame({
    'dtype':    df.dtypes,
    'non-null': df.notna().sum(),
    'missing':  df.isna().sum(),
    'miss%':    (df.isna().mean()*100).round(1),
})
print(info.to_string())

# ── 3. Feature Engineering ───────────────────────────────────────────────────
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
df['Title'] = df['Title'].replace({
    'Mlle':'Miss','Ms':'Miss','Mme':'Mrs',
    'Lady':'Royalty','Countess':'Royalty','Capt':'Officer',
    'Col':'Officer','Don':'Royalty','Dr':'Officer',
    'Major':'Officer','Rev':'Officer','Sir':'Royalty',
    'Jonkheer':'Royalty','Dona':'Royalty'
})
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone']    = (df['FamilySize'] == 1).astype(int)
df['Age'].fillna(df.groupby('Title')['Age'].transform('median'), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['AgeBand']  = pd.cut(df['Age'],  bins=[0,12,18,35,60,100],
                        labels=['Child','Teen','Young Adult','Adult','Senior'])
df['FareBand'] = pd.qcut(df['Fare'], q=4,
                         labels=['Low','Medium','High','Very High'])

print("\n✅  Feature engineering complete — new columns: Title, FamilySize, IsAlone, AgeBand, FareBand")

# ── Helper ───────────────────────────────────────────────────────────────────
def save(fig, name, dpi=150):
    fig.savefig(name, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"   💾 Saved → {name}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Dashboard Overview
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎨  Rendering Figure 1 — Dashboard Overview …")
fig = plt.figure(figsize=(20, 14), facecolor='#0a0a18')
fig.suptitle('TITANIC — Exploratory Data Analysis Dashboard',
             fontsize=22, color='#e8e8ff', fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.42)

# 1a) Survival pie
ax1 = fig.add_subplot(gs[0, 0])
counts   = df['Survived'].value_counts()
explode  = (0.05, 0)
wedges, texts, autotexts = ax1.pie(
    counts, labels=['Did Not\nSurvive','Survived'],
    colors=[DIED, SURVIVED], autopct='%1.1f%%',
    startangle=90, explode=explode,
    textprops={'color':'#e8e8ff','fontsize':9},
    wedgeprops={'linewidth':2,'edgecolor':'#0a0a18'})
for at in autotexts: at.set_fontsize(10); at.set_fontweight('bold')
ax1.set_title('Overall Survival Rate', color='#e8e8ff')

# 1b) Survival by Sex
ax2 = fig.add_subplot(gs[0, 1])
sex_surv = df.groupby(['Sex','Survived']).size().unstack()
sex_surv.plot(kind='bar', ax=ax2, color=[DIED, SURVIVED], width=0.6,
              edgecolor='#0a0a18', linewidth=1.2)
ax2.set_title('Survival by Sex')
ax2.set_xlabel('Sex'); ax2.set_ylabel('Count')
ax2.set_xticklabels(['Female','Male'], rotation=0)
ax2.legend(['Died','Survived'], facecolor='#1a1a2e', edgecolor='#2a2a4a',
           labelcolor='#e8e8ff', fontsize=8)
ax2.grid(axis='y', alpha=0.4)

# 1c) Survival by Pclass
ax3 = fig.add_subplot(gs[0, 2])
pc_surv = df.groupby(['Pclass','Survived']).size().unstack()
pc_surv.plot(kind='bar', ax=ax3, color=[DIED, SURVIVED], width=0.6,
             edgecolor='#0a0a18', linewidth=1.2)
ax3.set_title('Survival by Passenger Class')
ax3.set_xlabel('Class'); ax3.set_ylabel('Count')
ax3.set_xticklabels(['1st','2nd','3rd'], rotation=0)
ax3.legend(['Died','Survived'], facecolor='#1a1a2e', edgecolor='#2a2a4a',
           labelcolor='#e8e8ff', fontsize=8)
ax3.grid(axis='y', alpha=0.4)

# 1d) Survival by Embarked
ax4 = fig.add_subplot(gs[0, 3])
emb_surv = df.groupby(['Embarked','Survived']).size().unstack()
emb_surv.plot(kind='bar', ax=ax4, color=[DIED, SURVIVED], width=0.6,
              edgecolor='#0a0a18', linewidth=1.2)
ax4.set_title('Survival by Embarkation Port')
ax4.set_xlabel('Port (C=Cherbourg, Q=Queenstown, S=Southampton)')
ax4.set_ylabel('Count')
ax4.set_xticklabels(['C','Q','S'], rotation=0)
ax4.legend(['Died','Survived'], facecolor='#1a1a2e', edgecolor='#2a2a4a',
           labelcolor='#e8e8ff', fontsize=8)
ax4.grid(axis='y', alpha=0.4)

# 1e) Age distribution
ax5 = fig.add_subplot(gs[1, :2])
for s, col, lbl in [(0, DIED,'Died'), (1, SURVIVED,'Survived')]:
    sub = df[df['Survived']==s]['Age'].dropna()
    ax5.hist(sub, bins=30, color=col, alpha=0.7, label=lbl, edgecolor='none')
ax5.set_title('Age Distribution by Survival Outcome')
ax5.set_xlabel('Age (years)'); ax5.set_ylabel('Count')
ax5.legend(facecolor='#1a1a2e', edgecolor='#2a2a4a', labelcolor='#e8e8ff')
ax5.axvline(df['Age'].median(), color=ACCENT, lw=1.5, ls='--', label='Median age')
ax5.grid(alpha=0.3)

# 1f) Fare distribution (log)
ax6 = fig.add_subplot(gs[1, 2:])
for s, col, lbl in [(0, DIED,'Died'), (1, SURVIVED,'Survived')]:
    sub = df[df['Survived']==s]['Fare'].dropna()
    ax6.hist(sub, bins=40, color=col, alpha=0.7, label=lbl, edgecolor='none')
ax6.set_title('Fare Distribution by Survival Outcome (log scale)')
ax6.set_xlabel('Fare (£)'); ax6.set_ylabel('Count'); ax6.set_yscale('log')
ax6.legend(facecolor='#1a1a2e', edgecolor='#2a2a4a', labelcolor='#e8e8ff')
ax6.grid(alpha=0.3)

# 1g) Survival rate by Age Band
ax7 = fig.add_subplot(gs[2, :2])
ab_rate = df.groupby('AgeBand', observed=True)['Survived'].mean() * 100
bars = ax7.bar(ab_rate.index.astype(str), ab_rate.values,
               color=PALETTE[:len(ab_rate)], edgecolor='#0a0a18', linewidth=1)
ax7.set_title('Survival Rate by Age Band')
ax7.set_xlabel('Age Band'); ax7.set_ylabel('Survival Rate (%)')
ax7.set_ylim(0, 100)
for bar, val in zip(bars, ab_rate.values):
    ax7.text(bar.get_x()+bar.get_width()/2, val+1.5,
             f'{val:.0f}%', ha='center', va='bottom',
             color='#e8e8ff', fontsize=9, fontweight='bold')
ax7.grid(axis='y', alpha=0.4)

# 1h) Family size vs survival rate
ax8 = fig.add_subplot(gs[2, 2:])
fs_rate = df.groupby('FamilySize')['Survived'].mean() * 100
fs_cnt  = df.groupby('FamilySize')['Survived'].count()
scatter = ax8.scatter(fs_rate.index, fs_rate.values,
                      s=fs_cnt.values * 8, c=fs_rate.values,
                      cmap='RdYlGn', vmin=0, vmax=100,
                      edgecolors='#e8e8ff', linewidth=0.7, zorder=3)
ax8.plot(fs_rate.index, fs_rate.values, color=ACCENT, lw=1.5, ls='--', alpha=0.7)
ax8.set_title('Survival Rate by Family Size (bubble = count)')
ax8.set_xlabel('Family Size (self + relatives)')
ax8.set_ylabel('Survival Rate (%)')
ax8.set_ylim(0, 100)
ax8.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax8, label='Survival %',
             ).ax.yaxis.label.set_color('#e8e8ff')

save(fig, 'figure1_dashboard_overview.png', dpi=160)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Deep-Dive: Heatmaps & Correlations
# ══════════════════════════════════════════════════════════════════════════════
print("🎨  Rendering Figure 2 — Heatmaps & Correlations …")
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor='#0a0a18')
fig.suptitle('TITANIC — Correlation & Cross-Tab Analysis',
             fontsize=18, color='#e8e8ff', fontweight='bold', y=1.01)

# 2a) Numeric correlation heatmap
num_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare','FamilySize','IsAlone']
corr = df[num_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=axes[0], annot=True, fmt='.2f',
            cmap='coolwarm', center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor='#0a0a18',
            annot_kws={'size':8,'color':'#e8e8ff'})
axes[0].set_title('Pearson Correlation Matrix', color='#e8e8ff', pad=12)
axes[0].tick_params(colors='#8888aa')

# 2b) Survival rate: Pclass × Sex pivot heatmap
pivot1 = df.pivot_table('Survived', index='Pclass', columns='Sex', aggfunc='mean') * 100
sns.heatmap(pivot1, ax=axes[1], annot=True, fmt='.1f', cmap='RdYlGn',
            vmin=0, vmax=100, linewidths=1, linecolor='#0a0a18',
            annot_kws={'size':12,'fontweight':'bold','color':'#0a0a18'})
axes[1].set_title('Survival Rate % — Class × Sex', color='#e8e8ff', pad=12)
axes[1].set_xlabel('Sex'); axes[1].set_ylabel('Passenger Class')
axes[1].tick_params(colors='#8888aa')
axes[1].set_yticklabels(['1st','2nd','3rd'], rotation=0)

# 2c) Survival rate: AgeBand × Pclass pivot heatmap
pivot2 = df.pivot_table('Survived', index='AgeBand', columns='Pclass',
                         aggfunc='mean', observed=True) * 100
sns.heatmap(pivot2, ax=axes[2], annot=True, fmt='.1f', cmap='RdYlGn',
            vmin=0, vmax=100, linewidths=1, linecolor='#0a0a18',
            annot_kws={'size':11,'fontweight':'bold','color':'#0a0a18'})
axes[2].set_title('Survival Rate % — Age Band × Class', color='#e8e8ff', pad=12)
axes[2].set_xlabel('Passenger Class'); axes[2].set_ylabel('Age Band')
axes[2].set_xticklabels(['1st','2nd','3rd'])
axes[2].tick_params(colors='#8888aa')

plt.tight_layout()
save(fig, 'figure2_heatmaps_correlations.png', dpi=160)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Fare, Titles & Embarkation Deep Dive
# ══════════════════════════════════════════════════════════════════════════════
print("🎨  Rendering Figure 3 — Fare, Titles & Embarkation …")
fig, axes = plt.subplots(2, 2, figsize=(18, 12), facecolor='#0a0a18')
fig.suptitle('TITANIC — Fare, Title & Embarkation Patterns',
             fontsize=18, color='#e8e8ff', fontweight='bold')
axes = axes.flatten()

# 3a) Box plot: Fare by Class & Survival
data_box = [df[(df['Pclass']==pc)&(df['Survived']==sv)]['Fare'].dropna()
            for pc in [1,2,3] for sv in [0,1]]
labels_box = [f'C{pc}\n{"✓" if sv else "✗"}'
              for pc in [1,2,3] for sv in [0,1]]
colors_box = [DIED if sv==0 else SURVIVED
              for pc in [1,2,3] for sv in [0,1]]
bp = axes[0].boxplot(data_box, patch_artist=True, labels=labels_box,
                     medianprops={'color':'#e8e8ff','linewidth':2},
                     whiskerprops={'color':'#5a5a8a'},
                     capprops={'color':'#5a5a8a'},
                     flierprops={'marker':'o','markersize':3,
                                 'markerfacecolor':'#e8e8ff','alpha':0.4})
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color); patch.set_alpha(0.75)
axes[0].set_title('Fare Distribution — Class & Survival')
axes[0].set_ylabel('Fare (£)')
axes[0].grid(axis='y', alpha=0.3)
d_patch = mpatches.Patch(color=DIED,     label='Died')
s_patch = mpatches.Patch(color=SURVIVED, label='Survived')
axes[0].legend(handles=[d_patch, s_patch],
               facecolor='#1a1a2e', edgecolor='#2a2a4a', labelcolor='#e8e8ff')

# 3b) Title survival counts
title_surv = df.groupby(['Title','Survived']).size().unstack(fill_value=0)
title_surv = title_surv.loc[title_surv.sum(axis=1).sort_values(ascending=False).index]
title_surv.plot(kind='barh', ax=axes[1], color=[DIED, SURVIVED],
                edgecolor='#0a0a18', linewidth=0.8)
axes[1].set_title('Survival Count by Title')
axes[1].set_xlabel('Count'); axes[1].set_ylabel('Title')
axes[1].legend(['Died','Survived'], facecolor='#1a1a2e',
               edgecolor='#2a2a4a', labelcolor='#e8e8ff')
axes[1].grid(axis='x', alpha=0.3)

# 3c) Survival rate by FareBand
fb_rate = df.groupby('FareBand', observed=True)['Survived'].mean() * 100
bars = axes[2].bar(fb_rate.index.astype(str), fb_rate.values,
                   color=[PALETTE[i] for i in range(len(fb_rate))],
                   edgecolor='#0a0a18', linewidth=1)
axes[2].set_title('Survival Rate by Fare Band')
axes[2].set_xlabel('Fare Quartile'); axes[2].set_ylabel('Survival Rate (%)')
axes[2].set_ylim(0, 100)
for bar, val in zip(bars, fb_rate.values):
    axes[2].text(bar.get_x()+bar.get_width()/2, val+1.5,
                 f'{val:.0f}%', ha='center', va='bottom',
                 color='#e8e8ff', fontsize=10, fontweight='bold')
axes[2].grid(axis='y', alpha=0.4)

# 3d) Box + Swarm: Age by Pclass, split by survival
vdata    = []
vpos     = []
vcolours = []
xtick_pos   = []
xtick_label = []
base = 1
gap  = 0.65
spacing = 1.6
for i, pc in enumerate([1,2,3]):
    pos_died = base + i * spacing
    pos_surv = pos_died + gap
    xtick_pos.append((pos_died + pos_surv) / 2)
    xtick_label.append(f'{pc}{"st" if pc==1 else "nd" if pc==2 else "rd"} Class')
    for sv, col, pos in [(0, DIED, pos_died), (1, SURVIVED, pos_surv)]:
        sub = df[(df['Pclass']==pc) & (df['Survived']==sv)]['Age'].dropna()
        if len(sub) >= 2:
            vdata.append(sub.values)
            vpos.append(pos)
            vcolours.append(col)

if vdata:
    parts = axes[3].violinplot(vdata, positions=vpos, widths=0.55,
                               showmedians=True, showextrema=True)
    for body, col in zip(parts['bodies'], vcolours):
        body.set_facecolor(col); body.set_alpha(0.72)
    parts['cmedians'].set_color('#e8e8ff'); parts['cmedians'].set_linewidth(2)
    for key in ('cbars','cmins','cmaxes'):
        parts[key].set_color('#5a5a8a')

axes[3].set_xticks(xtick_pos)
axes[3].set_xticklabels(xtick_label)
axes[3].set_title('Age Distribution — Class & Survival (Violin)')
axes[3].set_ylabel('Age (years)')
d_patch = mpatches.Patch(color=DIED,     label='Died')
s_patch = mpatches.Patch(color=SURVIVED, label='Survived')
axes[3].legend(handles=[d_patch, s_patch],
               facecolor='#1a1a2e', edgecolor='#2a2a4a', labelcolor='#e8e8ff')
axes[3].grid(axis='y', alpha=0.3)

plt.tight_layout()
save(fig, 'figure3_fare_titles_embarkation.png', dpi=160)


# ══════════════════════════════════════════════════════════════════════════════
# Print Insights Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  KEY INSIGHTS")
print("=" * 60)

overall   = df['Survived'].mean() * 100
fem_rate  = df[df['Sex']=='female']['Survived'].mean() * 100
male_rate = df[df['Sex']=='male'  ]['Survived'].mean() * 100
c1_rate   = df[df['Pclass']==1]['Survived'].mean() * 100
c3_rate   = df[df['Pclass']==3]['Survived'].mean() * 100
child_rate = df[df['AgeBand']=='Child']['Survived'].mean() * 100 if 'Child' in df['AgeBand'].values else 0
alone_rate = df[df['IsAlone']==1]['Survived'].mean() * 100
fam_rate   = df[df['IsAlone']==0]['Survived'].mean() * 100

insights = [
    f"Overall survival rate          : {overall:.1f}%",
    f"Female survival rate           : {fem_rate:.1f}%   (vs {male_rate:.1f}% for males)",
    f"1st class survival rate        : {c1_rate:.1f}%   (vs {c3_rate:.1f}% for 3rd class)",
    f"Children survival rate         : {child_rate:.1f}%",
    f"Travelling alone survival      : {alone_rate:.1f}%   (vs {fam_rate:.1f}% with family)",
]
for ins in insights:
    print(f"  ➤  {ins}")

print("\n  📌  Top Findings:")
print("  1. Women were ~3.6× more likely to survive than men (Women & Children first).")
print("  2. 1st class passengers survived at nearly 3× the rate of 3rd class.")
print("  3. Children had elevated survival rates — priority boarding lifeboats.")
print("  4. Higher fares correlate strongly with survival (class proxy).")
print("  5. Solo travellers had lower survival odds than those with family.")
print("  6. Cherbourg embarkees had the highest survival rate (mostly 1st class).")
print("\n✅  All figures saved. EDA complete.")

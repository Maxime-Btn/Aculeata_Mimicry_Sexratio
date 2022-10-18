### Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


### Colormap
cmap = mpl.colormaps['Blues_r']
cmap.set_under(color="black")

### Set the figure
fig, ax = plt.subplots()
fig.subplots_adjust(top=0.985,
                    bottom=0.075,
                    left=0.0,
                    right=1.0,
                    hspace=0.155,
                    wspace=0.22)

ax.set_xlabel(r'Survival advantage provided by the sting $\alpha$', size=20, fontweight='bold')
ax.set_ylabel(r'Cost of males on predator learning $\beta$', size=20, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=15)
ax.set_facecolor(color='black')

### Set the colorbar
norm = mpl.colors.Normalize(vmin=0, vmax=1)

cbar_ax_1 = fig.add_axes([0.80, 0.085, 0.02, 0.85])
cb1 = mpl.colorbar.ColorbarBase(cbar_ax_1, cmap=cmap, norm=norm, orientation='vertical')
cb1.ax.tick_params(labelsize=15)

cb1.ax.set_title('Frequency of \n persistence', size=15, fontweight='bold', y=1.01)

### Data
df = pd.read_csv("./data/df_one_no_mimicry_aB.csv")

lev = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1,
       1.01]

df['sr'] = df['M'] / (df['M'] + df['F'])

df['av_sp1'] = df['eq_sp1'].groupby([df['a'], df['B']]).transform('mean')
df['av_sr'] = df['sr'].groupby([df['a'], df['B']]).transform('mean')

### Plotting
aspect = df['a'].unique()[-1] / df['B'].unique()[-1]
ax.set_aspect(aspect)

ax.tricontourf(df['a'], df['B'], df['av_sp1'], levels=lev, cmap=cmap,
               vmin=0, vmax=1, alpha=1, antialiased=False)

cs = ax.tricontour(df['a'], df['B'], df['av_sp1'], levels=[0.35, 0.5, 0.65], colors=['lightblue', 'blue', 'darkblue'],
                   linestyles=':', vmin=0, vmax=1, alpha=1, antialiased=False, linewidths=3)

ax.clabel(cs, fontsize=15, inline_spacing=0.2)

plt.show()

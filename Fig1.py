### Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

### Colormap
cmap = mpl.colormaps['PuOr']
cmap.set_bad(color="black")

### Set the figure
fig, ax = plt.subplots(1, 2)
fig.subplots_adjust(top=1.0,
                    bottom=0.0,
                    left=0.055,
                    right=0.835,
                    hspace=0.267,
                    wspace=0.18)

ax = ax.flatten()

ax[0].set_title(r"(a) Investment in favour of sons ($h=2$)", fontweight='bold', fontsize=15)
ax[1].set_title(r"(b) Investment in favour of daughters ($h=5$)", fontweight='bold', fontsize=15)

### Set the colorbar
norm = mpl.colors.Normalize(vmin=0, vmax=1)

cbar_ax_1 = fig.add_axes([0.89, 0.10, 0.02, 0.77])
cb1 = mpl.colorbar.ColorbarBase(cbar_ax_1, cmap=cmap, norm=norm, orientation='vertical',
                                ticks=[0, 0.2, 0.5, 0.8, 1])
cb1.ax.set_yticklabels(['0', 'Female-biased', 'Equally balanced', 'Male-biased', '1'], fontsize=13.5)

cb1.ax.set_title('Proportion of male \n at equilibrium', fontsize=15, fontweight='bold')

### Data and plotting
df_name = 'df_one_no_mimicry_plk.csv'

lev = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
       1, 1.01]

for i in range(2):
    df_brut = pd.read_csv("./data/{0}".format(df_name))

    df = df_brut.loc[(df_brut['k1'] == [2, 5][i])]

    ax[i].set_xlabel(r'Predation rate $p$', fontsize=20, fontweight='bold')
    ax[i].set_ylabel(r'Defence level $\lambda$', fontsize=20, fontweight='bold')
    ax[i].tick_params(axis='both', which='major', labelsize=15)

    aspect = df_brut['p'].unique()[-1] / df_brut['l1'].unique()[-1]
    ax[i].set_aspect(aspect)
    ax[i].set_facecolor(color='black')

    df['sr'] = df['M'] / (df['M'] + df['F'])

    df = df.loc[df.eq_sp1 == 1]

    df['av_sr'] = df['sr'].groupby([df['p'], df['l1']]).transform('mean')

    ax[i].tricontourf(df['p'], df['l1'], df['av_sr'], levels=lev, cmap=cmap,
                      vmin=0, vmax=1, alpha=1, antialiased=True)

    cs = ax[i].tricontour(df['p'], df['l1'], df['av_sr'], levels=lev, colors=['darkred'],
                          linestyles=[(0, (5, 10))], vmin=0, vmax=1, alpha=1, antialiased=False, linewidths=2)

    ax[i].clabel(cs, fontsize=12, inline_spacing=0.2)

plt.show()

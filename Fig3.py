### Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec

lev = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1,
       1.01]

### Colormaps
cmap_blue = mpl.colors.LinearSegmentedColormap.from_list("", ["white", "#0C06F3"])
cmap_orange = mpl.colors.LinearSegmentedColormap.from_list("", ["white", "#F3891D"])
cmap_purple = mpl.colors.LinearSegmentedColormap.from_list("", ["white", "#690696"])
cmap_yellow = mpl.colors.LinearSegmentedColormap.from_list("", ["white", "#96897A"])

cmap_white = mpl.colors.LinearSegmentedColormap.from_list("", ['white', 'white'])
cmap_tot = mpl.colors.ListedColormap(["#0C06F3", "#690696", "#96897A", "#F3891D"])

cmap_sp1 = mpl.colors.LinearSegmentedColormap.from_list("", ["blue", "red"])
cmap_sp2 = mpl.colors.LinearSegmentedColormap.from_list("", ["blue", "yellow"])

cmap_sp1.set_under(color="black")
cmap_sp2.set_under(color="black")

### Set the figure
fig = plt.figure(figsize=(10, 5))
fig.subplots_adjust(top=0.94,
                    bottom=0.085,
                    left=0.04,
                    right=0.95)

gs = GridSpec(nrows=2, ncols=2, wspace=-0.2, hspace=0.3)
ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 1])

ax0.set_title('(a)', fontsize=15, fontweight='bold')
ax1.set_title('(b)', fontsize=15, fontweight='bold')
ax2.set_title('(c)', fontsize=15, fontweight='bold')

ax0.set_xlabel(r"Relative female noxiousness: $\lambda_2$-$\lambda_1$", y=0, fontsize=20, weight='bold')
ax0.set_ylabel(r"Relative investment in son production: $h_2$-$h_1$", fontsize=20, fontweight='bold')
ax0.tick_params(axis='both', which='major', labelsize=15)

ax1.set_xlabel(r'Defence level $\lambda_1$', fontsize=17, fontweight='bold')
ax1.set_ylabel(r'Defence level $\lambda_2$', fontsize=17, fontweight='bold')
ax1.tick_params(axis='both', which='major', labelsize=15)

ax2.set_xlabel(r'Investment in sons $h_1$', fontsize=17, fontweight='bold')
ax2.set_ylabel(r'Investment in sons $h_2$', fontsize=17, fontweight='bold')
ax2.tick_params(axis='both', which='major', labelsize=15)

### Set the colorbar
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cbar_ax_1 = fig.add_axes([0.89, 0.10, 0.02, 0.77])
cb1 = mpl.colorbar.ColorbarBase(cbar_ax_1, cmap=cmap_tot, norm=norm, orientation='vertical',
                                ticks=[0.125, 0.375, 0.625, 0.875])
cb1.ax.set_yticklabels(['Coextinction', 'Only species 1', 'Only species 2', 'Coexistence'], fontsize=13.5)

cb1.ax.set_title('State of the community\nat equilibrium', fontsize=15, fontweight='bold')


### Fig 5a - Data
df_brut = pd.read_csv("./data/df_two_no_mimicry.csv")
df_brut['l_diff'] = np.round(df_brut['l2'] - df_brut['l1'], 2)
df_brut['k_diff'] = df_brut['k2'] - df_brut['k1']

df = df_brut[['M', 'F', 'm', 'f', 'l_diff', 'k_diff', 'eq_sp1', 'eq_sp2']]

df['state'] = df['eq_sp1'].astype(str) + df['eq_sp2'].astype(str)

df_grouped = df.groupby([df['l_diff'], df['k_diff']])

df['binary_mode'] = df_grouped['state'].transform(lambda x: x.value_counts().index[0])
df['freq'] = df_grouped['state'].transform(lambda x: max(x.value_counts(normalize=True)))

df['blue'], df['purple'], df['yellow'], df['orange'] = 1, 1, 1, 1

df['blue'] = df['blue'].where(df['binary_mode'] == '00', 0)
df['purple'] = df['purple'].where(df['binary_mode'] == '10', 0)
df['yellow'] = df['yellow'].where(df['binary_mode'] == '01', 0)
df['orange'] = df['orange'].where(df['binary_mode'] == '11', 0)

### Fig 5a - Interpolation
x, y = df['l_diff'], df['k_diff']

xi = np.linspace(-0.04, 0.04, 100)
yi = np.linspace(-4, 4, 100)
xi, yi = np.meshgrid(xi, yi)

zi_blue = griddata((x, y), df['blue'], (xi, yi), method='cubic')
zi_orange = griddata((x, y), df['orange'], (xi, yi), method='cubic')
zi_purple = griddata((x, y), df['purple'], (xi, yi), method='cubic')
zi_yellow = griddata((x, y), df['yellow'], (xi, yi), method='cubic')

zi3 = griddata((x, y), df['freq'], (xi, yi), method='cubic')

zi3[zi3 < 0] = 0
zi3[zi3 > 1] = 1

zi3bis = 10 * (zi3 - 0.25) / 7.5
zi3bis[zi3bis < 0] = 0

zi_blue[zi_blue < 0] = 0
zi_blue[zi_blue > 1] = 1

zi_orange[zi_orange < 0] = 0
zi_orange[zi_orange > 1] = 1

zi_purple[zi_purple < 0] = 0
zi_purple[zi_purple > 1] = 1

zi_yellow[zi_yellow < 0] = 0
zi_yellow[zi_yellow > 1] = 1

extent = [-0.04, 0.04, -4, 4]

### Fig 5a - Plotting
ax0.imshow(zi_orange, vmin=0, vmax=1, origin='lower', extent=extent, aspect=0.01, cmap=cmap_orange,
           alpha=zi_orange)

cs_orange = ax0.contour(xi, yi, zi_orange, levels=[0.75], colors="#F3891D", linewidths=3, alpha=0.7,
                        linestyles=['dashed'])
cs_blue = ax0.contour(xi, yi, zi_blue, levels=[0.75], colors="#0C06F3", linewidths=3, alpha=0.7, linestyles=['dashed'])
cs_purple = ax0.contour(xi, yi, zi_purple, levels=[0.75], colors="#690696", linewidths=3, alpha=0.7,
                        linestyles=['dashed'])
cs_yellow = ax0.contour(xi, yi, zi_yellow, levels=[0.75], colors="#96897A", linewidths=3, alpha=0.7,
                        linestyles=['dashed'])

ax0.clabel(cs_orange, fontsize=15, inline_spacing=0.2)
ax0.clabel(cs_blue, fontsize=15, inline_spacing=0.2)
ax0.clabel(cs_purple, fontsize=15, inline_spacing=0.2)
ax0.clabel(cs_yellow, fontsize=15, inline_spacing=0.2)

ax0.imshow(zi_blue, vmin=0, vmax=1, origin='lower', extent=extent, aspect=0.01, cmap=cmap_blue, alpha=zi_blue)
ax0.imshow(zi_purple, vmin=0, vmax=1, origin='lower', extent=extent, aspect=0.01, cmap=cmap_purple, alpha=zi_purple)
ax0.imshow(zi_yellow, vmin=0, vmax=1, origin='lower', extent=extent, aspect=0.01, cmap=cmap_yellow, alpha=zi_yellow)
ax0.imshow(zi3bis, vmin=0, vmax=1, origin='lower', extent=extent, aspect=0.01, cmap=cmap_white, alpha=1 - zi3bis)

### Fig 5b - Data and plotting
df = pd.read_csv("./data/df_two_no_mimicry_l1l2.csv")

aspect = df_brut['l1'].unique()[-1] / df_brut['l2'].unique()[-1]
ax1.set_aspect(aspect)

ax1.set_facecolor(color='black')

df['av_sp1'] = df['eq_sp1'].groupby([df['l1'], df['l2']]).transform('mean')
df['av_sp2'] = df['eq_sp2'].groupby([df['l1'], df['l2']]).transform('mean')

ax1.tricontourf(df['l1'], df['l2'], df['av_sp1'], levels=lev, cmap=cmap_sp1,
                vmin=0, vmax=1, alpha=1, antialiased=True)

ax1.tricontourf(df['l1'], df['l2'], df['av_sp2'], levels=lev, cmap=cmap_sp2,
                vmin=0, vmax=1, alpha=0.6, antialiased=True)

cs1 = ax1.tricontour(df['l1'], df['l2'], df['av_sp1'], levels=[0.5], colors=['black'],
                     linestyles=[(0, (5, 10))], vmin=0, vmax=1, alpha=1, antialiased=False, linewidths=2)

cs2 = ax1.tricontour(df['l1'], df['l2'], df['av_sp2'], levels=[0.5], colors=['white'],
                     linestyles=[(0, (1, 10))], vmin=0, vmax=1, alpha=1, antialiased=False, linewidths=2)

ax1.clabel(cs1, fontsize=15, inline_spacing=0.2)
ax1.clabel(cs2, fontsize=15, inline_spacing=0.2)

### Fig 5c - Data and plotting
df = pd.read_csv("./data/df_two_no_mimicry_k1k2.csv")

aspect = df_brut['k1'].unique()[-1] / df_brut['k2'].unique()[-1]
ax2.set_aspect(aspect)

ax2.set_facecolor(color='black')

df['av_sp1'] = df['eq_sp1'].groupby([df['k1'], df['k2']]).transform('mean')
df['av_sp2'] = df['eq_sp2'].groupby([df['k1'], df['k2']]).transform('mean')

ax2.tricontourf(df['k1'], df['k2'], df['av_sp1'], levels=lev, cmap=cmap_sp1,
                vmin=0, vmax=1, alpha=1, antialiased=True)

ax2.tricontourf(df['k1'], df['k2'], df['av_sp2'], levels=lev, cmap=cmap_sp2,
                vmin=0, vmax=1, alpha=0.6, antialiased=True)

cs1 = ax2.tricontour(df['k1'], df['k2'], df['av_sp1'], levels=[0.5], colors=['black'],
                     linestyles=[(0, (5, 10))], vmin=0, vmax=1, alpha=1, antialiased=False, linewidths=2)

cs2 = ax2.tricontour(df['k1'], df['k2'], df['av_sp2'], levels=[0.5], colors=['white'],
                     linestyles=[(0, (1, 10))], vmin=0, vmax=1, alpha=1, antialiased=False, linewidths=2)

ax2.clabel(cs1, fontsize=15, inline_spacing=0.2)
ax2.clabel(cs2, fontsize=15, inline_spacing=0.2)

plt.show()

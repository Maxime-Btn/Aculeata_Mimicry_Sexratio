### Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import griddata

### Colormaps
cmap_blue = mpl.colors.LinearSegmentedColormap.from_list("", ["white","#0C06F3"])
cmap_orange = mpl.colors.LinearSegmentedColormap.from_list("", ["white","#F3891D"])
cmap_purple = mpl.colors.LinearSegmentedColormap.from_list("", ["white","#690696"])
cmap_yellow = mpl.colors.LinearSegmentedColormap.from_list("", ["white","#96897A"])

cmap_white = mpl.colors.LinearSegmentedColormap.from_list("", ['white','white'])

cmap_tot = mpl.colors.ListedColormap(["#0C06F3", "#690696", "#96897A", "#F3891D"])

### Set the figure
fig, ax = plt.subplots()
fig.subplots_adjust(top=0.98,
                    bottom=0.07,
                    left=0.0,
                    right=0.965,
                    hspace=0.165,
                    wspace=0.0)

ax.set_xlabel(r"Relative female noxiousness: $\lambda_2$-$\lambda_1$",y=0, fontsize=20, fontweight='bold')
ax.set_ylabel(r"Relative investment in sons: $h_2$-$h_1$", fontsize=20, fontweight='bold')
ax.tick_params(axis='both', which='major', labelsize=15)

### Set the colorbar
cbar_ax = fig.add_axes([0.80, 0.10, 0.02, 0.8])
cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap_tot, orientation='vertical', norm=mpl.colors.Normalize(0, 1),
                               ticks=[0.125,0.375,0.625,0.875])
cb.ax.set_yticklabels(['coextinction', 'only species 1\n(monomorphic)', 'only species 2\n(dimorphic)', 'coexistence'],
                      fontsize=13.5)

cb.ax.set_title('State of the community \n at equilibrium', fontsize=15, fontweight='bold')

### Data
df_brut = pd.read_csv("./data/df_two_dslm.csv")

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

### Interpolation
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

zi3 = 10 * (zi3 - 0.25) / 7.5
zi3[zi3 < 0] = 0

zi_blue[zi_blue < 0] = 0
zi_blue[zi_blue > 1] = 1

zi_orange[zi_orange < 0] = 0
zi_orange[zi_orange > 1] = 1

zi_purple[zi_purple < 0] = 0
zi_purple[zi_purple > 1] = 1

zi_yellow[zi_yellow < 0] = 0
zi_yellow[zi_yellow > 1] = 1

extent = [-0.04, 0.04, -4, 4]

### Plotting
cs_orange = ax.contour(xi,yi,zi_orange, levels=[0.75], colors="#F3891D",linewidths=3, alpha=0.5, linestyles=['dashed'])
cs_blue = ax.contour(xi,yi, zi_blue, levels=[0.75], colors="#0C06F3", linewidths=3, alpha=0.5, linestyles=['dashed'])
cs_purple = ax.contour(xi,yi, zi_purple, levels=[0.75], colors="#690696",linewidths=3, alpha=0.5, linestyles=['dashed'])
cs_yellow = ax.contour(xi,yi, zi_yellow, levels=[0.75], colors="#96897A",linewidths=3, alpha=0.5, linestyles=['dashed'])

ax.clabel(cs_orange, fontsize=15, inline_spacing=0.2)
ax.clabel(cs_blue, fontsize=15, inline_spacing=0.2)
ax.clabel(cs_purple, fontsize=15, inline_spacing=0.2)
ax.clabel(cs_yellow, fontsize=15, inline_spacing=0.2)

ax.imshow(zi_orange, vmin=0, vmax=1, origin='lower', extent=extent, aspect=0.01, cmap=cmap_orange,alpha=zi_orange)
ax.imshow(zi_blue, vmin=0, vmax=1, origin='lower', extent=extent, aspect=0.01, cmap=cmap_blue, alpha=zi_blue)
ax.imshow(zi_purple, vmin=0, vmax=1, origin='lower', extent=extent, aspect=0.01, cmap=cmap_purple,alpha=zi_purple)
ax.imshow(zi_yellow, vmin=0, vmax=1, origin='lower', extent=extent, aspect=0.01, cmap=cmap_yellow,alpha=zi_yellow)
ax.imshow(zi3, vmin=0, vmax=1, origin='lower', extent=extent, aspect=0.01, cmap=cmap_white,alpha=1 - zi3)

plt.show()

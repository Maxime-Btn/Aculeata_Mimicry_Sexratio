### Libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

### Colormap
cmap = mpl.colormaps['Blues_r']
cmap.set_under(color="black")

### Set the figure
fig, ax = plt.subplots(1, 2)
fig.subplots_adjust(top=1.0,
                    bottom=0.0,
                    left=0.053,
                    right=0.862,
                    hspace=0.257,
                    wspace=0.175)
ax = ax.flatten()

ax[0].set_title("Non-mimetic community (a)", fontweight='bold', fontsize=15)
ax[1].set_title(" Mimetic community (b)", fontweight='bold', fontsize=15)

### Set the colorbar
norm = mpl.colors.Normalize(vmin=0, vmax=1)
cbar_ax_1 = fig.add_axes([0.93, 0.10, 0.02, 0.77])
cb1 = mpl.colorbar.ColorbarBase(cbar_ax_1, cmap=cmap, norm=norm, orientation='vertical')
cb1.ax.tick_params(labelsize=15)

cb1.ax.set_title('Frequency of \n coexistence', fontsize=15, fontweight='bold', y=1.01)

### Data and plotting
df_name = ['df_two_no_mimicry_lk.csv', 'df_two_mimicry_lk.csv']

lev = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1,
       1.01]

for i in range(2):
    df = pd.read_csv("./data/{0}".format(df_name[i]))

    ax[i].set_xlabel(r'Female noxiousness: $\lambda_1$=$\lambda_2$', fontsize=20, fontweight='bold')
    ax[i].set_ylabel(r'Investment in sons: $h_1$=$h_2$', fontsize=20, fontweight='bold')
    ax[i].tick_params(axis='both', which='major', labelsize=15)

    aspect = df['l1'].unique()[-1] / df['k1'].unique()[-1]
    ax[i].set_aspect(aspect)

    ax[i].set_facecolor(color='black')

    df['av_coex'] = df['coexistence'].groupby([df['l1'], df['k1']]).transform('mean')

    ax[i].tricontourf(df['l1'], df['k1'], df['av_coex'], levels=lev, cmap=cmap,
                      vmin=0, vmax=1, alpha=1, antialiased=True)

    cs = ax[i].tricontour(df['l1'], df['k1'], df['av_coex'], levels=[0.25, 0.5, 0.75], colors=['white', 'red', 'black'],
                          linestyles=[(0, (5, 10)), 'solid', (0, (5, 10))], vmin=0, vmax=1, alpha=1, antialiased=False,
                          linewidths=[1, 3, 1])

    ax[i].clabel(cs, fontsize=15, inline_spacing=0.2)

plt.show()

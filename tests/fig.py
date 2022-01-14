import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
n_reviews = 50000
alpha_min = 1
alpha_max = 2.2

parameter = "metric"
title = "weighted_Vader"
loss_type = 'MAE'
x = ['cosine', 'euclidian']
y = [0.9416049372930906, 0.9433846936043706]
alpha = [1.44, 1.44]

# plt.bar(x,y,color=[cm.viridis((a-alpha_min)/(alpha_max-alpha_min)) for a in alpha], alpha=0.7)
# plt.xticks(fontsize= 20)
# plt.yticks(fontsize= 16)
# plt.ylabel(f'{loss_type} loss (star)', fontsize=20)
# plt.title(title, fontsize=20)
# # plt.show()
# plt.savefig(f"tests/fig/{n_reviews}/{parameter}_{title}_{loss_type}.svg")
# plt.clf()

fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=alpha_min, vmax=alpha_max)
fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap, alpha=0.7),
             cax=ax, orientation='horizontal', label='sentiment parameter \u03B1')
plt.savefig(f"tests/fig/{n_reviews}/alpha_colorbar.svg")
plt.clf()
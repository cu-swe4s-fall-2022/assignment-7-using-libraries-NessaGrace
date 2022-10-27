import matplotlib.pyplot as plt
#plt.rcParams.update({'font.size': 22})
import pandas as pd

# load in data set and assign column names
iris = pd.read_csv('iris.data', header = None)
iris.columns = ['sepal_width', 'sepal_length', 'petal_width', 'petal_length',
                'iris_species']

# produce boxplot of all measurements across all iris species
measurement_names = ['sepal_width', 'sepal_length', 'petal_width',
                     'petal_length']
plt.boxplot(iris[measurement_names], labels = measurement_names)
plt.ylabel('cm')
plt.savefig('iris_boxplot.png')
plt.show()

# produce scatterplot of petal length vs. petal width for all iris species
for species_name in set(iris['iris_species']):
    iris_subset = iris[iris['iris_species'] == species_name]
    plt.scatter(iris_subset['petal_width'], iris_subset['petal_length'],
                label = species_name, s = 5)
plt.legend()
plt.xlabel('petal_width (cm)')
plt.ylabel('petal_length (cm)')
plt.savefig('petal_length_v_width_scatter.png')
plt.show()

# produce both boxplot and scatterplot side by side
fig, axes = plt.subplots(2,2)
#fig.set_size_inches = ((10,10))
fig.delaxes(axes[1,0])
fig.delaxes(axes[1,1])

# left boxplot
axes[0,0].boxplot(iris[measurement_names], labels = measurement_names)
axes[0,0].set_xticklabels(measurement_names, fontsize=6)
axes[0,0].set_ylabel('cm', fontsize=8)
axes[0,0].tick_params(axis = 'y', labelsize=8)

# right scatterplot
for species_name in set(iris['iris_species']):
    iris_subset = iris[iris['iris_species'] == species_name]
    axes[0,1].scatter(iris_subset['petal_width'], iris_subset['petal_length'],
                label = species_name, s = 5, alpha = 0.5)
axes[0,1].legend(loc = 'upper left', prop = {'size': 6})
axes[0,1].set_xlabel('petal_width (cm)', fontsize = 8)
axes[0,1].set_ylabel('petal_length (cm)', fontsize = 8)
axes[0,1].tick_params(axis = 'x', labelsize = 8)
axes[0,1].tick_params(axis = 'y', labelsize = 8)

# remove top and right borders of plots
for i in range(2):
    for j in range(2):
        axes[i,j].spines['top'].set_visible(False)
        axes[i,j].spines['right'].set_visible(False)
        axes[i,j].spines['bottom'].set_visible(True)
        axes[i,j].spines['left'].set_visible(True)


plt.savefig('multi_panel_figure.png')
plt.show()

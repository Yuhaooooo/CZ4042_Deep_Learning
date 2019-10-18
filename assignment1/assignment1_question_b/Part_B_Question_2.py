import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns

csv_file = os.path.join('others', 'admission_predict.csv')

df = pd.read_csv(csv_file, index_col=[0])

corr = df.corr()

ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)

plt.savefig(os.path.join("others", "plot", "q2", "correlationMap.png"))

max_corr_extent=0
max_corr_index=(-1,-1)
for i in range(len(corr)-1):
    for j in range(i+1, len(corr)-1):
        if corr.iloc[i,j] > max_corr_extent:
            max_corr_extent = corr.iloc[i,j]
            max_corr_index=(i,j)
            

print('\n\n', '-'*10, '  Question 2(a)  ', '-'*10, '\n\n')
print('most correlated features: {}, {}\nthe correlation score is: {}'.format(corr.columns[max_corr_index[0]], corr.columns[max_corr_index[1]], max_corr_extent))

print('\n\n', '-'*10, '  Question 2(b)  ', '-'*10, '\n\n')
related_feature_of_admit = [(corr.columns[i], corr.iloc[-1, i]) for i in (np.argsort(corr.iloc[-1,:-1].tolist())[::-1])]
print('The feature with the highest correlations with the chances of admit: ', related_feature_of_admit[0])
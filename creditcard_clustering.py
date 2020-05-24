# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
data=pd.read_csv('creditcard_data.csv')
data_raw=data.copy()
#%%
'''
check out feature types
'''
data.dtypes
#%% 
'''
Missing values treatment
'''
pd.isnull(data).sum()

#'CREDIT_LIMIT': drop that row directly since it only has 1 missing value
data['CREDIT_LIMIT'].describe()
missing_index=data[data['CREDIT_LIMIT'].isnull()].index.to_list()
data=data.drop(index=missing_index[0])

# 'MINIMUM_PAYMENTS': fill missing values with median value (since it has outliers)
sns.kdeplot(data['MINIMUM_PAYMENTS'], shade=True) # kernel density estimation
plt.title('Kernel Density Estimation Plot') # positively skewed distribution
plt.savefig('KDE_MINIMUM_PAYMENTS.png')
data['MINIMUM_PAYMENTS']=data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].median())

pd.isnull(data).sum()

#%% 
'''
Outliers Treatment
'''
# calculate z-score
from scipy import stats
# drop string feature and features with meaningful range
data1=data.drop(columns=['CUST_ID', 'TENURE']) 

z_score=pd.DataFrame(np.abs(stats.zscore(data1)), columns=data1.columns) 

# Find out features with more than 2% outliers (absolute z-score >3)
z_score3=[]
over3_index=[] 
for i in z_score.columns:
    indexs=z_score.index[z_score[i] > 3].tolist()
    ans=i, "{:.3f}".format(len(indexs)/len(z_score)), indexs
    z_score3.append(ans) 
    if len(indexs)/len(z_score) > 0.02:
        over3_index.append(i)  

# remove 'BALANCE' and 'CASH_ADVANCE' since thay are regarded as high
# discriminative features
del over3_index[0]
del over3_index[1]

# create new feature to decrease impact of outliers
# square root value of 'BALANCE_FREQUENCY','CASH_ADVANCE_FREQUENCY',
# 'PURCHASES_TRX'
for i in over3_index:
    data1['sqrt_%s' % i]=data1[i].apply(np.sqrt)

#%%
'''
Create interaction features

we can understand clients' purchase level or preference
by knowing their average purchase amount, however we only got 
frequency for oneoff and installment purchase or cash advance for this dataset.
Frequency data is reagarded as metric of number of transaction since 
number of transaction divided by some value will be frequency.
(the higher frequency is, the less average purchase amount is)

Below are the formulas:
Average one-off purchase amount='ONEOFF_PURCHASES'/'ONEOFF_PURCHASES_FREQUENCY'
Average installment purchase amount='INSTALLMENTS_PURCHASES'/'PURCHASES_INSTALLMENTS_FREQUENCY'
Average can advance amount='CASH_ADVANCE'/'CASH_ADVANCE_TRX'

'''
data1['avg_oneoff_purchases']=data1['ONEOFF_PURCHASES']/data1['ONEOFF_PURCHASES_FREQUENCY']
data1['avg_oneoff_purchases']=data1['avg_oneoff_purchases'].fillna(0)

data1['avg_installment_purchases']=data1['INSTALLMENTS_PURCHASES']/data1['PURCHASES_INSTALLMENTS_FREQUENCY']
data1['avg_installment_purchases']=data1['avg_installment_purchases'].fillna(0)

data1['avg_cash_advance']=data1['CASH_ADVANCE']/data1['CASH_ADVANCE_TRX']
data1['avg_cash_advance']=data1['avg_cash_advance'].fillna(0)

data1.shape #(8949, 22)

#%%
'''
Digitize features into 8 ranges except 'TENURE' (due to its meaningful own range)
'''
import math
digit_index=list(data1.columns)

for i in digit_index:
    max_v=math.ceil(data1[i].describe()['max'])
    min_v=math.floor(data1[i].describe()['min'])
    bins_range=np.arange(min_v, max_v, (max_v-min_v)/8)    
    data1['digit_%s' % i]=np.digitize(data1[i], bins=bins_range)
    #print(np.unique(data1['digit_%s' % i], return_counts=True))

data1['CUST_ID']=data['CUST_ID']
data1['TENURE']=data['TENURE']

data1.shape #(8949, 46)

#%% 
'''
Correlation Matrix
Note:
To avoid the top and bottom boxes are cut off=> downgrade matplotlib
=>conda install -c conda-forge matplotlib=3.1.2
'''
corr_coef=data[1:].corr()

plt.figure(figsize=(25, 25))
sns.heatmap(corr_coef, cmap='Blues', annot=True, annot_kws={'size':14},
            xticklabels=corr_coef.columns,
            yticklabels=corr_coef.columns)
plt.title('Correlation Matrix')
plt.savefig('corr_matrix.png', dpi=300)

# find out feature pairs whose coefficient >= 0.7
corr_cols=corr_coef.columns.to_list()

signif_corr=[]
for i in range(len(corr_cols)):
    col=corr_cols[i]
    signif_corr.append(abs(corr_coef[col])[abs(corr_coef[col]) >= 0.7])
signif_corr_df=pd.DataFrame(signif_corr)
#signif_corr_df['PURCHASES']['ONEOFF_PURCHASES'] # 0.9168436510438295

#%%
# KDE (kernel density estimation) plot
sns.kdeplot(data1['PURCHASES_INSTALLMENTS_FREQUENCY'], shade=True)
sns.kdeplot(data1['ONEOFF_PURCHASES_FREQUENCY'], shade=True)
sns.kdeplot(data1['PURCHASES_FREQUENCY'], shade=True)
plt.title('Kernel Density Estimation Plot')
plt.savefig('kde_frequency.png', dpi=300)

sns.kdeplot(data1['INSTALLMENTS_PURCHASES'], shade=True)
sns.kdeplot(data1['ONEOFF_PURCHASES'], shade=True)
sns.kdeplot(data1['PURCHASES'], shade=True)
plt.title('Kernel Density Estimation Plot')
plt.savefig('kde_purchases.png', dpi=300)

'''
High Correlation Coefficient Pairs Analysis:
    
1. PURCHASES & ONEOFFPURCHASES: 0.92
When people use one-off purchases, purchase amount is higher than using 
installment purchases.

PURCHASES : Amount of purchases made from account
ONEOFFPURCHASES : Maximum purchase amount done in one-go

2. PURCHASESFREQUENCY & PURCHASESINSTALLMENTSFREQUENCY: 0.86
More people use installment purchases.

PURCHASESFREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased) 
PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)

3.CASHADVANCEFREQUENCY & CASHADVANCETRX: 0.80

CASHADVANCEFREQUENCY : How frequently the cash in advance being paid
CASHADVANCETRX : Number of Transactions made with "Cash in Advanced"
'''
#%%
'''
KDE plot for each feature
'''
data_dist=data.iloc[:,1:17]
data_columns=data_dist.columns

r,c=0,0
fig, axes=plt.subplots(4,4, figsize=(20,16))
#plt.tight_layout()
for i in data_columns:
    sns.distplot(data[i], ax=axes[r,c])
    c += 1
    if c == 4: 
        r += 1
        c=0
    if r == 4: break
plt.suptitle('Kernel Density Estimation Plot', fontsize=15)
plt.savefig('distplots.png', dpi=300)          
#%% 
'''
Modeling
'''
from sklearn.preprocessing import Normalizer, RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import GridSearchCV

'''
design pipelines to select numerical and categorical features
'''
from sklearn.base import BaseEstimator, TransformerMixin

# Since sklearn cannot directly handle DataFrames, we need to define a function 
# totransform it into numpy
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names] 

categ=list(data1.columns)[22:44]
categ1=np.delete(categ,[1, 9, 11]).tolist()
numer=['TENURE']

numerical_pipeline=Pipeline([('selector', DataFrameSelector(numer)),
                             ('RobustScaler', RobustScaler())])

categorical_pipeline=Pipeline([('selector', DataFrameSelector(categ1)),
                             ('OneHotEncoder', OneHotEncoder())])

selector_pipeline=FeatureUnion([('numerical_pipeline', numerical_pipeline),
                                ('categorical_pipeline', categorical_pipeline)])

#%%
'''
Pipelines
'''
categ_copy=categ1.copy()
categ_copy.append('TENURE')
data_model=data1[categ_copy]

# evaluation metric: silhouette score
def silhouette_score_cal(estimator,data):       
    preprocess=FeatureUnion([('selector_pipeline', selector_pipeline), 
                             ('Normalizer', Normalizer(norm='l2')),
                             ('pca', PCA(n_components=15))])            
    trans_results=preprocess.fit_transform(data)          
    clusters=estimator.fit_predict(data)
    score=silhouette_score(trans_results, clusters)
    return score

preprocess=FeatureUnion([('selector_pipeline', selector_pipeline), 
                             ('Normalizer', Normalizer(norm='l2')),
                             ('pca', PCA(n_components=15))])

trans_results=preprocess.fit_transform(data_model)  # for visualization    
kmeans=Pipeline([('preprocess', preprocess), ('kmeans', KMeans())])     
search_space=[{'kmeans__n_clusters':np.arange(3,10)}] # test various(3-9) n_clusters
cv = [(slice(None), slice(None))]

gs=GridSearchCV(estimator=kmeans,param_grid=search_space, 
                scoring=silhouette_score_cal,cv=cv, n_jobs=-1)

best_model=gs.fit(data_model)

#%%
# Best model results
best_model.best_params_
best_model.best_score_
grid_predict=best_model.predict(data_model)
data1['cluster']=grid_predict
grid_results=best_model.cv_results_
#%%
# Silhouette Score under Various Number of Cluster
grid_scores=grid_results['mean_test_score']
plt.plot(range(3,10), grid_results['mean_test_score'])
plt.title('Silhouette Score under Various Number of Cluster')
plt.savefig('silhouette_score_grid310c_20f_15pca.png', dpi=300)

#%%
# Save data
data_model.to_csv('data_model.csv')
data1.to_csv('data1.csv')

#%%
# visualize scatter plot
df_visual=pd.DataFrame(TruncatedSVD(n_components=2).fit_transform(trans_results), columns=['p1','p2'])

plt.figure(figsize=(10,8))
plt.scatter(df_visual['p1'], df_visual['p2'], c=grid_predict, cmap=plt.cm.summer)
plt.title('Clustering Results Visualization')
plt.savefig('clusters_visual.png', dpi=300)

#%%
# number of observations in each cluster
np.unique(grid_predict, return_counts=True) #[4518, 2828, 1603]

# histogram for each cluster
for feature in list(data.columns[1:]):
    g=sns.FacetGrid(data1, col='cluster')
    g=g.map(plt.hist, feature)
    plt.savefig('each_cluster_%s.png' % feature, dpi=300)

# mean value of each cluster
data1.insert(0, 'TENURE', data1['TENURE'], allow_duplicates=True) # just for convience
each_cluster=data1.groupby('cluster').mean().iloc[:, :20]

#%%
'''
Render Plot
split features into 3 parts based on value reange (low, high, top)
'''
high=['BALANCE', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
      'MINIMUM_PAYMENTS']
low=['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY', 
     'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'CASH_ADVANCE_TRX', 
     'PURCHASES_TRX','PRC_FULL_PAYMENT', 'TENURE']
top=['CREDIT_LIMIT', 'PURCHASES', 'PAYMENTS']

each_cluster_high=each_cluster[high].T
each_cluster_low=each_cluster[low].T
each_cluster_top=each_cluster[top].T

#%%
'''
Render Plot
feature value of each cluster
'''
def render_plot(data, dataLenth, labels, color, facecolor):    
    angles = np.linspace(0, 2*np.pi, dataLenth, endpoint=False)
    data = np.concatenate((data, [data[0]])) # for visualization
    angles = np.concatenate((angles, [angles[0]])) # for visualization
        
    ax = fig.add_subplot(121, polar=True)# polar: drawing circle
    ax.plot(angles, data, color, linewidth=1)
    ax.fill(angles, data, facecolor=facecolor, alpha=0.1)# fill color
    ax.set_thetagrids(angles * 180/np.pi, labels, fontproperties="SimHei")
    ax.set_title("Feature Value of Each Cluster", va='baseline', fontproperties="SimHei")
    ax.grid(True)

fig=plt.figure()
fig = plt.figure(figsize=(15,15))
labels = np.array(list(each_cluster_high.index))
render_plot(each_cluster_high.iloc[:,0], len(each_cluster_high.iloc[:,0]), 
            labels,'go-', 'g')
render_plot(each_cluster_high.iloc[:,1], len(each_cluster_high.iloc[:,0]), 
            labels,'bo-', 'b')
render_plot(each_cluster_high.iloc[:,2], len(each_cluster_high.iloc[:,0]), 
            labels,'ro-', 'r')
plt.savefig('render_high.png',dpi=300, bbox_inches ='tight')
plt.show()

fig = plt.figure(figsize=(15,15))
labels = np.array(list(each_cluster_low.index))
render_plot(each_cluster_low.iloc[:,0], len(each_cluster_low.iloc[:,0]), 
            labels,'go-', 'g')
render_plot(each_cluster_low.iloc[:,1], len(each_cluster_low.iloc[:,0]), 
            labels,'bo-', 'b')
render_plot(each_cluster_low.iloc[:,2], len(each_cluster_low.iloc[:,0]), 
            labels,'ro-', 'r')
plt.savefig('render_low.png', dpi=300, bbox_inches='tight')
plt.show()

fig = plt.figure(figsize=(15,15))
labels = np.array(list(each_cluster_top.index))
render_plot(each_cluster_top.iloc[:,0], len(each_cluster_top.iloc[:,0]), 
            labels,'go-', 'g')
render_plot(each_cluster_top.iloc[:,1], len(each_cluster_top.iloc[:,0]), 
            labels,'bo-', 'b')
render_plot(each_cluster_top.iloc[:,2], len(each_cluster_top.iloc[:,0]), 
            labels,'ro-', 'r')
plt.savefig('render_top.png', dpi=300, bbox_inches='tight')
plt.show()

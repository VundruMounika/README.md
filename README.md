# README.md
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


dataset=pd.read_csv(r"E:\Projects\ResponsiveApp\2255872-anime_data (1).csv")


# In[3]:


dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.eps.describe()


# In[6]:


dataset.columns


# In[7]:


dataset[(dataset['eps']> 24)& (dataset.duration.isna())].shape


# In[214]:


dataset_excluding_out = dataset[dataset['eps'] < 50]


# In[215]:


dataset_excluding_out[ 'eps_brackets' ] = pd.cut(dataset_excluding_out['eps'], bins = [1,10,20,30,40,50],\
                                             labels= [ 'cats1' , 'cats2' , 'cat3' , 'cat4' , 'cats'])


# In[ ]:





# In[216]:


dataset_excluding_out.shape


# In[217]:


dataset_excluding_out.groupby(['eps_brackets']).duration.mean()


# In[218]:


dataset[(dataset['eps']<24)&(dataset.duration.isna())].describe()


# In[219]:


dataset.isna().sum()


# In[220]:


dataset.describe().T


# In[223]:


dataset.drop(columns=['title','description'],axis=1,inplace=True)


# In[224]:


dataset.head()


# In[225]:


dataset.rating.describe()


# In[226]:


dataset.dropna(inplace=True)
dataset.shape


# In[227]:


12000-7465


# In[228]:


def continuos_univariate_analysis(data,
                                  feature,
                                  figsize=(12,8),
                                  kde=False,
                                  bins=None):
    f1,(ax_box,
        ax_hist)=plt.subplots(nrows=2,
                             sharex=True,
                             gridspec_kw={'height_ratios':(0.25,0.75)},
                             figsize=figsize)
    sns.color_palette("viridis", as_cmap=True)
    sns.boxplot(data=data,
               x=feature,
               ax=ax_box,
               showmeans=True,
               color='yellow')
    sns.histplot(data=data,
                 x=feature,
                 ax=ax_hist,
                 showmeans=True,
                 color='crest',
                 bins=bins,
                kde=kde) if bins else sns.histplot(
                    data=data, x=feature, ax=ax_hist, kde=kde, color='blue')
    ax_hist.axvline(data[feature].mean(), color='cyan', linestyle='--')
    ax_hist.axvline(data[feature].median(), color='orange', linestyle="-")


# In[229]:


def discrete_univariate_analysis(data, feature, perc=False, n=None):
    total=len(data[feature])
    count= data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1 ,5))
    else:
        plt.figure(figsize=(n + 1, 5))
    plt.xticks(rotation=90, fontsize=15)
    ax=sns.countplot(
        data=data,
        x=feature,
        palette="flare",
        order=data[feature].value_counts().index[:n].sort_values(
            ascending=False))
    for p in ax.patches:
        if perc == True:
            label="{:.1f}%".format(100 * p.get_height() / total)
        else:
              label=p.get_height()
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(label, (x,y),
                    ha="center",
                    va="center",
                    size=12,
                    xytext=(0, 5),
                    textcoords="offset points")
    plt.show()


# In[230]:


dataset.columns


# In[231]:


continuos_univariate_analysis(dataset, 'rating')


# In[232]:


continuos_univariate_analysis(dataset, 'duration')


# In[233]:


dataset[dataset['duration' ] >=80]['rating'].mean()


# In[234]:


dataset[dataset['duration'] >=100]['rating'].mean()


# In[235]:


dataset[dataset['duration'] >=110]['rating'].mean()


# In[236]:


dataset[(dataset['duration'] >=5) & (dataset['duration']<=30)]['rating'].mean()


# In[237]:


discrete_univariate_analysis(dataset, "ongoing", perc=True)


# In[238]:


dataset[dataset['ongoing'] == True]['rating'].mean()


# In[239]:


dataset[dataset['ongoing'] == True]['duration'].mean()


# In[240]:


discrete_univariate_analysis(dataset, "sznOfRelease", perc=True)


# In[241]:


discrete_univariate_analysis(dataset, "studio_primary", perc=True)


# In[242]:


dataset[dataset['rating'] > 4]['studio_primary'].value_counts(normalize=True).mul(100).round(2)


# In[243]:


discrete_univariate_analysis(dataset, 'contentWarn', perc=True)


# In[244]:


corr_cols = [item for item in dataset.columns if "tag" not in item]


# In[245]:


corr_cols


# In[246]:


plt.figure(figsize=(16,7))
sns.heatmap(dataset[corr_cols].corr(), annot=True, vmin = -1, vmax= 1, fmt=' .2f', cmap='viridis')
plt.show()


# In[247]:


dataset.drop(columns= ['eps','watched'], inplace=True)


# In[248]:


dataset.shape


# In[249]:


plt.figure(figsize=(15,8))
sns.boxplot(x = 'sznOfRelease', y='rating', data=dataset )


# In[250]:


x = dataset.drop(['rating'], axis=1)
y = dataset['rating']


# In[251]:


x.info()


# In[252]:


x = pd.get_dummies(x, columns=x.select_dtypes(include=['object', 'category']).columns.tolist(),drop_first=True)
x.head()


# In[253]:


x.drop(columns='ongoing', inplace=True)

x.info()
# In[254]:


x.info()


# In[ ]:






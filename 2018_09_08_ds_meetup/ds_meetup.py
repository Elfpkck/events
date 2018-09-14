#!/usr/bin/env python
# coding: utf-8

# # Pipeline, Feature Union, Custom Transformer

# **Plan:**
# * Pipeline
# * Feature Union
# * Custom Transformers

# ## Pipeline

# ![alt text](pipeline.jpg)

# ![alt text](conveyor.jpg)

# ### Advantages
# * Code organization
# * Metrics of whole pipeline
# * Easier deployment

# ### Disadvantages:
# * Less flexibility

# ![alt text](pipeline_versus_featureunion.jpg)

# ## Transformer

# ![alt text](optimus.jpg)

# `Transformers` are for pre-processing before modeling.  
# `Estimators` (`Models`) are used to make predictions.

# ## Two ways to create pipeline

# ### 1. `Pipeline()` class

# In[1]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

pipe_long = Pipeline([
    ("scaler", MinMaxScaler()), 
    ("svm", SVC())
])
pipe_long.steps


# ### 2. `make_pipeline()` function

# In[5]:


from sklearn.pipeline import make_pipeline

pipe_short = make_pipeline(MinMaxScaler(), SVC())
pipe_short = make_pipeline(MinMaxScaler(), MinMaxScaler(), SVC())  # name of duplicates
pipe_short.steps


# **Names in make_pipeline are taken from types of classes in lowercase**

# In[3]:


type(MinMaxScaler()), type(SVC())


# ## First simple transformer

# **Downloading iris dataset**

# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

pd.options.mode.chained_assignment = None  # hide unnecessary Pandas notifications

iris = load_iris()
df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
df.head()


# ![alt text](http://s5047.pcdn.co/wp-content/uploads/2015/04/iris_petal_sepal.png)

# ![alt text](http://images.myshared.ru/5/447185/slide_5.jpg)

# **What if I don't know how to write classes?**

# In[7]:


def make_perfect_array(x):
    x[:,:] *= 0
    return x


# In[9]:


from sklearn.preprocessing import FunctionTransformer

pipe = make_pipeline(
    MinMaxScaler(), 
    FunctionTransformer(make_perfect_array)
)


# In[10]:


pipe.fit_transform(iris.data)


# **The same without pipeline**

# In[11]:


data = MinMaxScaler().fit_transform(iris.data)
FunctionTransformer(make_perfect_array).fit_transform(data)


# **The problem:** `FunctionTransformer` works only with numpy arrays :(  
# So we have to write a class to work with Pandas df

# **Downloading newsgroups dataset**

# In[12]:


from sklearn.datasets import fetch_20newsgroups

fetch_20newsgroups()["target_names"]


# In[13]:


categories = [
    'rec.autos',
    'sci.med',
    'talk.politics.guns',
]
news_train = fetch_20newsgroups(subset='train', categories=categories)
news_test = fetch_20newsgroups(subset='test', categories=categories)


# In[14]:


df = pd.DataFrame(data= np.c_[news_train['data'], news_train['target']],
                  columns=['text','target'])
df.head()


# In[15]:


df.text.str.len().plot(title="Text Length")


# In[16]:


df.text = df.text.str.slice(0, 5000)
df.text.str.len().plot(title="Text Length")


# ## Pipeline organization

# #### 1. If you want to use pipeline as estimator (`predict` method)

# All classes, except last, have to have method `transform`. The last one has to have `predict` method.

# **Necessary methods:**
# * `fit`
#     * all classes
# * `transform`
#     * all except last
# * `predict`
#     * the last one

# #### 2. If you want to use pipline as transformer (`transform` method)

# **Necessary methods:**
# * `fit`
#     * all classes
# * `transform`
#     * all
# * `predict`
#     * nowhere (if don't call predict())

# In[17]:


from sklearn.base import BaseEstimator, TransformerMixin

class CutTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, stop: int):
        self.stop = stop

    def fit(self, x, y=None):
        return self
    
    def transform(self, df_x, df_y=None):
        df_x.loc[:, "text"] = df_x.text.str.slice(0, self.stop)
        return df_x


# In[18]:


class AutoTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, df_x, df_y=None):
        ar = df_x.text.str.count("auto").astype(float).values
        return ar.reshape(-1, 1)
    
class MedTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, df_x, df_y=None):
        ar = df_x.text.str.count("med").astype(float).values
        return ar.reshape(-1, 1)
    
class GunTransformer(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    
    def transform(self, df_x, df_y=None):
        ar = df_x.text.str.count("gun").astype(float).values
        return ar.reshape(-1, 1)


# ## FeatureUnion Perfomance

# In[19]:


from sklearn.pipeline import FeatureUnion, make_union

union_ = make_union(AutoTransformer(), MedTransformer(), GunTransformer())
df_big = pd.concat([df for _ in range(100)])  # creation of big dataframe


# In[20]:


get_ipython().run_cell_magic('timeit', '', 'union_.fit_transform(df_big)')


# In[21]:


a = AutoTransformer()
m = MedTransformer()
g = GunTransformer()


# In[22]:


get_ipython().run_cell_magic('timeit', '', 'a.fit_transform(df_big) \nm.fit_transform(df_big) \ng.fit_transform(df_big) ')


# **Another transformer**

# In[23]:


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name: list):
        self.name = name
        
    def fit(self, *args, **kwargs):
        return self
    
    def transform(self, x, y=None):
        return pd.DataFrame(x, columns=self.name)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler

union = [("auto", AutoTransformer()), ("med", MedTransformer()), ("gun", GunTransformer())]
names = [name for name, _ in union]

pipe = make_pipeline(
    CutTransformer(5000), 
    FeatureUnion([
        ("auto", AutoTransformer()), 
        ("med", MedTransformer()), 
        ("gun", GunTransformer())
    ]),
    MinMaxScaler(),
    DataFrameTransformer(names),
    RandomForestClassifier(random_state=42)
)


# In[25]:


pipe.steps


# ![alt text](pipeline_.png)

# ## Pipleine usage and possibility to change steps

# In[27]:


from copy import copy

pipe_ = copy(pipe)
pipe_.fit(df.drop("target", axis=1), df.target)
pipe_.steps[0] = ('cuttransformer', CutTransformer(3000))
pipe_.predict(df)


# **Pipeline parameters**

# In[28]:


pipe.get_params().keys()


# **Creating of test dataframe**

# In[29]:


df_test = pd.DataFrame(data=np.c_[news_test['data'], news_test['target']],
                       columns=['text','target'])


# **Recreating of dataframe just in case**

# In[30]:


df = pd.DataFrame(data= np.c_[news_train['data'], news_train['target']],
                  columns=['text','target'])


# ## Pipeline metrics

# In[31]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'cuttransformer__stop': [5000, 6000, 7000],
    'randomforestclassifier__max_depth': [4, 5],
    'randomforestclassifier__max_features': [1, 2, 3],
    'randomforestclassifier__n_estimators': [43, 44, 45]
}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(df.drop("target", axis=1), df.target)

print(f"Best params:\n{grid.best_params_}\n")
print(f"Best score: {grid.best_score_:.4f}")
print(f"Score on testset: {grid.score(df_test.drop('target', axis=1), df_test.target):.4f}")


# ### Wrong metrics without pipeline

# In[32]:


df = pd.DataFrame(data= np.c_[news_train['data'], news_train['target']],
                  columns=['text','target'])
y_train = df.target


# In[33]:


df = CutTransformer(6000).fit_transform(df)


# **The same that 3 transformers do :)**

# In[34]:


cols = ["auto", "med", "gun"]
for col in cols:
    df[col] = df.text.str.count(col)


# In[35]:


x_train = MinMaxScaler().fit_transform(df[cols])


# In[36]:


from sklearn.model_selection import cross_val_score

param = {
    'max_depth': 5,
    'max_features': 1,
    'n_estimators': 43,
    'random_state': 42
}

score = cross_val_score(RandomForestClassifier(**param), x_train, y_train, cv=5).mean()
print(f"{score:.4f}")


# ## Any combinations of `pipeline` and `feature union`

# ![SegmentLocal](deeper.gif)

# **Result:**
# * Pipeline
# * Feature Union
# * Custom Transformers

# In[37]:


get_ipython().system(u'jupyter nbconvert --to script ds_meetup.ipynb')


# # Thank you

# In[ ]:





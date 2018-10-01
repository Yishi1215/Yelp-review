
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
from langdetect import detect

from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook, show
output_notebook()


df= pd.read_csv('cleaned_data_v1_full.csv',index_col=False)



alpha = 10**np.linspace(10,-2,100)*0.5


def lr(X_train, X_test, y_train, y_test):
    regr_cv = LinearRegression()
    lr = regr_cv.fit(X_train, y_train)
    return lr.score(X_test, y_test)

def ridge(X_train, X_test, y_train, y_test):
    regr_cv = RidgeCV(alphas=alpha, normalize=True)
    ridge = regr_cv.fit(X_train, y_train)
    return ridge.score(X_test, y_test)

def lasso(X_train, X_test, y_train, y_test):
    regr_cv = LassoCV(alphas=alpha, normalize=True)
    lasso = regr_cv.fit(X_train, y_train)
    return lasso.score(X_test, y_test)


def rf(X_train, X_test, y_train, y_test):
    regr_cv = RandomForestRegressor(n_estimators=10)
    rf = regr_cv.fit(X_train, y_train)
    return rf.score(X_test, y_test)



def gb(X_train, X_test, y_train, y_test):
    gb_cv = GradientBoostingRegressor()
    gb = gb_cv.fit(X_train, y_train)
    return gb.score(X_test, y_test)


# ## CountVectorizer：Only use text data to predict

# In[49]:


lr_rsqure=[]
ridge_rsqure=[]
lasso_rsqure=[]
rf_rsqure=[]
gb_rsqure=[]
for mindf in range(1,10):
    count_vect = CountVectorizer(min_df=mindf)
    X = count_vect.fit_transform(df['text'])
    X_train, X_test, y_train, y_test = train_test_split(X, df['useful'], test_size=0.33, random_state=42)
    lr_rsqure.append(lr(X_train, X_test, y_train, y_test))
    ridge_rsqure.append(ridge(X_train, X_test, y_train, y_test))
    lasso_rsqure.append(lasso(X_train, X_test, y_train, y_test))
    rf_rsqure.append(rf(X_train, X_test, y_train, y_test))
    gb_rsqure.append(rf(X_train, X_test, y_train, y_test))


# In[50]:


p1 = figure(plot_width=800, plot_height=400)
p1.xaxis.axis_label = 'Min_df'
p1.yaxis.axis_label = 'Coefficient of determination'
p1.line(range(1,10), lr_rsqure, color='firebrick', legend='Linear Regression')
p1.line(range(1,10), ridge_rsqure, color='navy', legend='Ridge')
p1.line(range(1,10), lasso_rsqure, color='olive', legend='Lasso')
p1.line(range(1,10), rf_rsqure, color='orange', legend='Random Forest')
p1.line(range(1,10), gb_rsqure, color='fuchsia', legend='Gradient Boosting')
p1.legend.location = "bottom_left"
output_file("text_CountVectorizer_coefficient_of_determination.html")


# ## TfidfVectorizer：Only use text data to predict

# In[51]:


lr_rsqure=[]
ridge_rsqure=[]
lasso_rsqure=[]
rf_rsqure=[]
gb_rsqure=[]
for mindf in range(1,10):
    count_vect = TfidfVectorizer(min_df=mindf, ngram_range=(1,3))
    X = count_vect.fit_transform(df['text'])
    X_train, X_test, y_train, y_test = train_test_split(X, df['useful'], test_size=0.33, random_state=42)
    lr_rsqure.append(lr(X_train, X_test, y_train, y_test))
    ridge_rsqure.append(ridge(X_train, X_test, y_train, y_test))
    lasso_rsqure.append(lasso(X_train, X_test, y_train, y_test))
    rf_rsqure.append(rf(X_train, X_test, y_train, y_test))
    gb_rsqure.append(rf(X_train, X_test, y_train, y_test))


# In[52]:


p2 = figure(plot_width=800, plot_height=400)
p2.xaxis.axis_label = 'Min_df'
p2.yaxis.axis_label = 'Coefficient of determination'
p2.line(range(1,10), lr_rsqure, color='firebrick', legend='Linear Regression')
p2.line(range(1,10), ridge_rsqure, color='navy', legend='Ridge')
p2.line(range(1,10), lasso_rsqure, color='olive', legend='Lasso')
p2.line(range(1,10), rf_rsqure, color='orange', legend='Random Forest')
p2.line(range(1,10), gb_rsqure, color='fuchsia', legend='Gradient Boosting')
p2.legend.location = "bottom_left"
output_file("text_TfidfVectorizer_coefficient_of_determination.html")


# ## Only use numeric data to predict

# In[53]:


X = df[['stars', 'user_review_count', 'friends', 'user_total_useful', 'total_funny',
       'total_cool', 'user_average_stars', 'business_stars',
       'business_review_count', 'days', 'text_count', 'pol',
       'user_avg_useful']]
y = df['useful']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
methods = ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 'Gradient Boosting']
score = []
score.append(lr(X_train, X_test, y_train, y_test))
score.append(ridge(X_train, X_test, y_train, y_test))
score.append(lasso(X_train, X_test, y_train, y_test))
score.append(rf(X_train, X_test, y_train, y_test))
score.append(gb(X_train, X_test, y_train, y_test))

mothod_score = zip(methods, score)
[print('{:4} Score: {}'.format(*pair)) for pair in mothod_score];


# In[54]:


p3 = figure(x_range=methods, plot_height=250, title="Only use numeric data to predict the review usefulness ",
           toolbar_location=None, tools="")
p3.vbar(x=methods, top=score, width=0.3)
output_file("numeric_data_coefficient_of_determination.html")


# feature_importances

# In[55]:


# Random Forest
regr_cv = RandomForestRegressor(n_estimators=30)
rf = regr_cv.fit(X_train, y_train)
print(rf.score(X_test, y_test))

print(rf.feature_importances_)
feature_list = list(X.columns)
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[56]:


# Gradient Boosting
my_imputer = Imputer()
imputed_X = my_imputer.fit_transform(X)
clf = GradientBoostingRegressor()
clf.fit(imputed_X, y)

feature_list = list(X.columns)
# Get numerical feature importances
importances = list(clf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)[0:5]
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[58]:


f, i = zip(*feature_importances)
fig,axs = plot_partial_dependence(clf,       
                                   features=[0, 1, 2, 3, 4], # column numbers of plots we want to show
                                   X=X,            # raw predictors data.
                                   feature_names= list(f), # labels on graphs
                                   grid_resolution=10) # number of values to plot on x axis
fig.savefig('plot_partial_dependence.png')


# ## Combine text and numeric data

# In[95]:


df_ml = df[['stars', 'text', 'useful', 'user_review_count', 'friends', 'user_total_useful', 'total_funny', 
            'total_cool', 'user_average_stars', 'business_stars','business_review_count', 'days', 'text_count', 
            'pol', 'user_avg_useful']]

# Obtain the text data: get_text_data
get_text_data = FunctionTransformer(lambda x: x['text'], validate=False)

# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[['stars', 'user_review_count', 'friends', 'user_total_useful', 
                                                    'total_funny', 'total_cool', 'user_average_stars', 
                                                    'business_stars','business_review_count', 'days', 
                                                    'text_count', 'pol', 'user_avg_useful']], validate=False)



# Fit and transform the text data: just_text_data
just_text_data = get_text_data.fit_transform(df_ml)

# Fit and transform the numeric data: just_numeric_data
just_numeric_data = get_numeric_data.fit_transform(df_ml)

# Split using ALL data in sample_df
X_train, X_test, y_train, y_test = train_test_split(df_ml[['stars', 'text', 'useful', 'user_review_count', 
                                                           'friends', 'user_total_useful', 'total_funny', 
                                                           'total_cool', 'user_average_stars', 'business_stars',
                                                           'business_review_count', 'days', 'text_count', 'pol', 
                                                           'user_avg_useful']], 
                                                    df_ml['useful'] , 
                                                    random_state=22)


# CountVectorizer

# In[96]:


lr_pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', LinearRegression())
    ])

ridge_pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RidgeCV(alphas=alpha, normalize=True))
    ])

lasso_pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', LassoCV(alphas=alpha, normalize=True))
    ])

rf_pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestRegressor(random_state = 42))
    ])

gb_pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', CountVectorizer())
                ]))
             ]
        )),
        ('clf', GradientBoostingRegressor())
    ])

min_df = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ngram_range=[(1, 2), (2, 3)]
param_grid = {'union__text_features__vectorizer__min_df': min_df,
              'union__text_features__vectorizer__ngram_range': ngram_range}
n_estimators = range(10, 200, 10)
param_grid_rf = {'union__text_features__vectorizer__min_df': min_df,
                 'union__text_features__vectorizer__ngram_range': ngram_range,
                 'clf__n_estimators': n_estimators}


# In[ ]:


grid = GridSearchCV(lr_pipe, cv=3, param_grid=param_grid)
grid.fit(X_train,y_train)
print("Linear Regression Best: %f using %s" % (grid.best_score_, grid.best_params_))

grid = GridSearchCV(ridge_pipe, cv=3, param_grid=param_grid)
grid.fit(X_train,y_train)
print("Ridge Best: %f using %s" % (grid.best_score_, grid.best_params_))

grid = GridSearchCV(lasso_pipe, cv=3, param_grid=param_grid)
grid.fit(X_train,y_train)
print("Lasso Best: %f using %s" % (grid.best_score_, grid.best_params_))

grid = GridSearchCV(rf_pipe, cv=3, param_grid=param_grid_rf)
grid.fit(X_train,y_train)
print("Random Forest Best: %f using %s" % (grid.best_score_, grid.best_params_))

grid = GridSearchCV(gb_pipe, cv=3, param_grid=param_grid_rf)
grid.fit(X_train,y_train)
print("Gradient Boosting Best: %f using %s" % (grid.best_score_, grid.best_params_))


# TfidfVectorizer

# In[ ]:


lr_pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer())
                ]))
             ]
        )),
        ('clf', LinearRegression())
    ])

ridge_pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer())
                ]))
             ]
        )),
        ('clf', RidgeCV(alphas=alpha, normalize=True))
    ])

lasso_pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer())
                ]))
             ]
        )),
        ('clf', LassoCV(alphas=alpha, normalize=True))
    ])

rf_pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer())
                ]))
             ]
        )),
        ('clf', RandomForestRegressor(random_state = 42))
    ])

gb_pipe = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer())
                ]))
             ]
        )),
        ('clf', GradientBoostingRegressor())
    ])


# In[ ]:


min_df = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
ngram_range=[(1, 2), (2, 3)]
param_grid = {'union__text_features__vectorizer__min_df': min_df,
              'union__text_features__vectorizer__ngram_range': ngram_range}
n_estimators = range(10, 200, 10)
param_grid_rf = {'union__text_features__vectorizer__min_df': min_df,
                 'union__text_features__vectorizer__ngram_range': ngram_range,
                 'clf__n_estimators': n_estimators}

grid = GridSearchCV(lr_pipe, cv=3, param_grid=param_grid)
grid.fit(X_train,y_train)
print("Linear Regression Best: %f using %s" % (grid.best_score_, grid.best_params_))

grid = GridSearchCV(ridge_pipe, cv=3, param_grid=param_grid)
grid.fit(X_train,y_train)
print("Ridge Best: %f using %s" % (grid.best_score_, grid.best_params_))

grid = GridSearchCV(lasso_pipe, cv=3, param_grid=param_grid)
grid.fit(X_train,y_train)
print("Lasso Best: %f using %s" % (grid.best_score_, grid.best_params_))

grid = GridSearchCV(rf_pipe, cv=3, param_grid=param_grid_rf)
grid.fit(X_train,y_train)
print("Random Forest Best: %f using %s" % (grid.best_score_, grid.best_params_))

grid = GridSearchCV(gb_pipe, cv=3, param_grid=param_grid_rf)
grid.fit(X_train,y_train)
print("Gradient Boosting Best: %f using %s" % (grid.best_score_, grid.best_params_))


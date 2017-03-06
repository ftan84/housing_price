#' ---
#' title: Greater Seattle Area Housing--Sales Price Prediction
#' author: Francis Tan
#' date: 2017-02-27
#' abstract: |
#'     The goal of this project is to predict the sale price of a property
#'     by employing various predictive machine learning models in an ensemble
#'     given housing data such as the number of bedrooms/bathrooms, square
#'     footage, year built as well as other less intuitive variables as
#'     provided by the Zillow API.
#' header-includes:
#'     - \usepackage{booktabs}
#'     - \usepackage{longtable}
#' ---

#' # Introduction
#' The Greater Seattle Area housing market has gone through a dramatic price
#' increase in recent years with the median valuation at $609,100, an 11.3%
#' increase from last year and a 4.7% increase forecasted for the next
#' year[^zillowstat]. Because of the dramatic change that has occurred in a
#' short period of time, it has become increasingly difficult to predict
#' property values. We believe that a machine learning model can predict a
#' property value with reasonable accuracy given the property features and
#' the time aspect (_lastSoldDate_).

#' [^zillowstat]: Source: Zillow Data as of February 2017.

#+ echo=False
import math
import numpy as np
import pandas as pd
import pprint
import pweave
from sklearn import linear_model, svm, tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
import sqlite3

pd.set_option('display.max_columns', None)

#' # Data Collection and Preprocessing
#' ## Data Collection Process
#' The most important element of any data science project is the data itself.
#' This project heavily utilizes data from Zillow, a digital real estate
#' destination for buyers, sellers, and agents. Fortunately, Zillow provides
#' a public API which provides a convenience to an otherwise tedious task.
#' Although the availability of a public API has made the data collection
#' process simple, there are some limitations that we had to be cognizant of.
#' Our vision was to start with a "seed" property which in turn would
#' collect "comps" or comparables. Comps are simply other properties that
#' have similar features to our seed property. This will provide a buyer an
#' idea of what the value of the property should be.
#' 
#' The first limitation is that the full set of information that we were
#' looking for cannot be extracted from one API endpoint. Zillow does not
#' provide an endpoint which returns property information of comps given a
#' seed property. What it provides instead is one endpoint that returns a
#' list of comp property IDs (Zillow Property ID or ZPID) given a seed
#' property address and a separate endpoint that returns property information
#' given a ZPID. Furthermore, the comp endpoint returns a maximum of 25 comps
#' per seed property. Thus the collection process is divided into three steps:

#' 1. Collect comp IDs given a property address using _GetDeepSearchResults_.
#' 2. Loop through each ZPID, collect 25 more comps for each, and append
#'    results to list of the other ZPIDs.
#' 3. Collect property information for each ZPID collected using
#'    _GetDeepComps_.

#' The second limitation is that Zillow has limited the number of calls
#' allowed per day to 1000. This poses a problem if one's intent was to
#' collect a significant amount of data. This limits our collection process
#' further since we had to resort to making two calls. A simple solution was
#' to include a sleep timer of 24 hours when a call encounters a rate limit
#' warning. Although somewhat inconvenient, the solution achieved what we
#' needed to accomplish.
#' 
#' ## Training Data
#' The table below is just a sample of what the training data looks like.
#' We've removed many of the columns to make sure the table fits in the page.
#' This is only to provide an idea of the formatting.

#+ echo=False
conn = sqlite3.connect('zillowdata.db')
q = '''SELECT *
       FROM properties
       WHERE lastSoldPrice IS NOT NULL'''
training_raw = pd.read_sql_query(q, conn)
conn.close()

#+ echo=False, results='tex', caption='Sample of dataset'
print(training_raw.head(10).to_latex(
    columns=[
        'street',
        'city',
        'zip',
        'finishedSqFt',
        'lastSoldDate',
        'lastSoldPrice'
    ],
    index=False
))

#' Printing the _shape_ attribute shows that we have 2826 observations and 23
#' columns.

#+ term=True
training_raw.shape

#' Finally, we have the following columns

#+ term=True
training_raw.dtypes

#' Since the goal of this project is to predict the sale price, it is obvious
#' that the _lastSoldPrice_ should be the response variable while the other
#' columns can act as feature variables. Of course, some processing such as
#' dummy variable conversion is required before training begins.
#' 
#' ## Data Processing
#' 
#' The next step is to process and clean the data. First let's take a look at
#' each variable and decide which ones need to be excluded. ZPID and street
#' address logically do not affect sales price and thus can be excluded. Street
#' address may explain some sale price variabilty, however it requires further
#' processing for proper formatting, that is, we must eliminate unit numbers,
#' suffix standardization (Dr. vs Drive), etc. This proves to be a difficult
#' task that is beyond the scope of this project. Further, the effects of this
#' variable is closely related to region. We have chosen to exclude it here
#' but may be worth exploring further in the future. Finally, the state
#' variable can also be excluded here as we are keeping the scope of this
#' project to WA only.

#+ term=True
training = training_raw  # Save original data intact
training.drop(['zpid', 'street', 'state'], axis=1, inplace=True)
training.dtypes

#' We can see that many of these variables are of type _object_. We'll need to
#' convert these to the appropriate types. Most of these columns, excluding
#' date columns, can be converted to numeric.

cols = training.columns[training.columns.isin([
    'taxAssessmentYear',
    'taxAssessment',
    'yearBuilt',
    'lotSizeSqFt',
    'finishedSqFt',
    'bathrooms',
    'bedrooms',
    'lastSoldPrice',
    'zestimate',
    'zestimateValueChange',
    'zestimateValueLow',
    'zestimateValueHigh',
    'zestimatePercentile'
])]
for col in cols:
    training[col] = pd.to_numeric(training[col])

#' Now let's convert _lastSoldDate_ and _zestimateLastUpdated_ to dates.

cols = training.columns[training.columns.isin([
    'lastSoldDate',
    'zestimateLastUpdated'
])]
for col in cols:
    training[col] = pd.to_datetime(training[col], infer_datetime_format=True)

#' One problem with the _datetime_ data type is that it will not work with
#' scikit-learn. So we need to convert this to a numerical type. One way to
#' resolve this is to separate the date columns into year, month, and day.

training = training.assign(
    lastSoldDateYear=pd.DatetimeIndex(training['lastSoldDate']).year,
    lastSoldDateMonth=pd.DatetimeIndex(training['lastSoldDate']).month,
    lastSoldDateDay=pd.DatetimeIndex(training['lastSoldDate']).day,
    zestimateLastUpdatedYear=pd.DatetimeIndex(
        training['zestimateLastUpdated']
    ).year,
    zestimateLastUpdatedMonth=pd.DatetimeIndex(
        training['zestimateLastUpdated']
    ).month,
    zestimateLastUpdatedDay=pd.DatetimeIndex(
        training['zestimateLastUpdated']
    ).day
)
training.drop(['lastSoldDate', 'zestimateLastUpdated'], axis=1, inplace=True)
training.dtypes

#' Next we need to see which of these variables need to be converted to factor
#' variables. City, state, zip, FIPScounty, useCode, and region all qualify.
#' One thing to caution when creating dummy variables is the number of unique
#' categories each column has. Large number of categories may be impractical
#' for this project as it requires a significant amount of computing resources.

#+ term=True
training['city'] = training['city'].astype('category')
training['city'].describe()
training['zip'] = training['zip'].astype('category')
training['zip'].describe()
training['FIPScounty'] = training['FIPScounty'].astype('category')
training['FIPScounty'].describe()
training['useCode'] = training['useCode'].astype('category')
training['useCode'].describe()
training['region'] = training['region'].astype('category')
training['region'].describe()

#' We can see that none of these variables have an unreasonably high number of
#' unique categories with the exception of region. It contains 147 categories
#' which may be too high, however, we will assume that our machine can handle
#' this for now.

#' Let's take a look at our columns now.

#+ term=True
training.dtypes

#' Before we convert the categorical columns to dummy variables, let's look
#' at the correlations compared to the sales price.

#+ term=True
training.corr()['lastSoldPrice'].sort_values(ascending=False, inplace=False)

#' As suspected, _zestimate_ columns are highly correlated to the sales price.
#' A _zestimate_ is essentially Zillow's predicted value. Since we are trying
#' to achieve the same thing in this project, let's not include Zillow's efforts
#' here.

training.drop(['zestimate', 'zestimateLastUpdatedYear',
               'zestimateLastUpdatedMonth', 'zestimateLastUpdatedDay',
               'zestimateValueChange', 'zestimateValueLow',
               'zestimateValueHigh', 'zestimatePercentile'],
              axis=1, inplace=True)

#' Here are the fininshed columns.

#+ term=True
training.dtypes

#+ echo=False, results='tex'
print(training.describe().to_latex(columns=[
    'taxAssessmentYear',
    'taxAssessment',
    'yearBuilt',
    'lotSizeSqFt',
    'finishedSqFt',
    'bathrooms']))

#+ echo=False, results='tex'
print(training.describe().to_latex(columns=['bedrooms', 'lastSoldPrice']))

#' We can see that the median price in our data set is $577,000 which is quite
#' high!

#' As it turns out, we have NaNs in our data as seen below:

print(training.isnull().any())

#' We need to remove these as NaNs will not work in the training process.

training.dropna(inplace=True)
print(training.isnull().any())

#' ## Dummy Variables

#' Finally, let's make the dummy variable conversion. This can easily be
#' achieved using the _get\_dummies_ function.

training = pd.get_dummies(training, columns=['city', 'zip', 'FIPScounty',
                                             'useCode', 'region'],
                          drop_first=True)
print(training.shape)

#' We have 245 columns as shown above, which we can verify by adding the number
#' of unique categories - 1 with the number of non-categorical columns.


#' # Training

#' Now that we have prepared the data, we can begin the training process. Since
#' we do not have new test data on hand, we will need to split a portion for
#' final evaluation. Let's set aside 20% of the data for just that. We
#' achieve this using scikit-learn's _train\_test\_split_ function.

#+ term=True
x_train, x_test, y_train, y_test = train_test_split(
    training.drop('lastSoldPrice', axis=1),
    training['lastSoldPrice'],
    test_size=0.2,
    random_state=1201980
)

#' The general plan here is to train several different models, make predictions
#' for each of those models, and use those predictions to train and predict
#' a separate ensemble model. In essence, we will have two layers of
#' training/prediction processes. Each model will be cross-validated with
#' 10-folds using the K-fold method and will utilize a grid-search to find the
#' best combination of parameters.

#' ## Ridge Regression

#' Ridge regression addresses some of the problems of Ordinary Least Squares by
#' imposing a penalty on the size of coefficients[^ridge]. We are also building
#' a grid of alphas that range from 0.1 to 10 in increments of 0.2. Since
#' we have a relatively small dataset, we can afford to build a large grid.

#' [^ridge]: Scikit-learn Documentation-Ridge Regressino (http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)

ridge = linear_model.Ridge()
param_grid = {
    'alpha': np.arange(0.1, 10, 0.1)
}
grid = GridSearchCV(ridge, param_grid, cv=10, n_jobs=-1,
                    scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
print('Ridge Regression:')
print('Best params: {}'.format(grid.best_params_))
print('RMSE: {}'.format(math.sqrt(abs(grid.best_score_))))

#' As we can see, the grid search has found that an alpha of about 7.4 produced
#' the best RMSE at 198433. This is not nearly as accurate as I would like it
#' to be but it is a good start.


#' ## Support Vector Machine

svm = svm.SVR()
param_grid = {
    'C': [0.0001, 0.001, 0.1, 1.0, 5]
}
grid = GridSearchCV(svm, param_grid, cv=10, n_jobs=-1,
                    scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
print('Support Vector Regressor:')
print('Best params: {}'.format(grid.best_params_))
print('RMSE: {}'.format(math.sqrt(abs(grid.best_score_))))


#' ## Lasso

#' Lasso is a generalized linear model that has a tendency to prefer solutions
#' with fewer parameter values[^lasso]. Our particular project isn't considered
#' a high dimensional problem and thus this model should be appropriate.

#' [^lasso]: Scikit-learn Documentation-Lasso (http://scikit-learn.org/stable/modules/linear_model.html#lasso)

lasso = linear_model.Lasso()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 5],
    'max_iter': [5000, 10000]
}
grid = GridSearchCV(lasso, param_grid, cv=10, n_jobs=-1,
                    scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
print('Lasso:')
print('Best params: {}'.format(grid.best_params_))
print('RMSE: {}'.format(math.sqrt(abs(grid.best_score_))))


#' ## Decision Tree

#' The Decision Tree model is one of the more simple and interpretable models.
#' We have chosen to include it here for its simplicity.

tree_reg = tree.DecisionTreeRegressor()
param_grid = {}
grid = GridSearchCV(tree_reg, param_grid, cv=10, n_jobs=-1,
                    scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
print('Decision Tree:')
print('RMSE: {}'.format(math.sqrt(abs(grid.best_score_))))


#' ## Elastic Net

#' Elastic Net is another linear model that has features of both Rdige
#' Regression and Lasso.

enet = linear_model.ElasticNet()
param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 5, 10],
    'l1_ratio': np.arange(0, 1, 0.1)
}
grid = GridSearchCV(enet, param_grid, cv=10, n_jobs=-1,
                    scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
print('Elastic Net:')
print('Best params: {}'.format(grid.best_params_))
print('RMSE: {}'.format(math.sqrt(abs(grid.best_score_))))


#' # Ensemble

#' After training multiple individual training models, we see that the results
#' are less than ideal. The best RMSE of 198433, comes from the Ridge
#' Regression model. That kind of RMSE is much too large for it to be practical
#' in real world settings. The below table shows the CV validated RMSEs for our
#' individual models.

#' | Model | RMSE |
#' |---|---|
#' | Ridge Regression | 198433 |
#' | Support Vector Regression | 526903 |
#' | Lasso | 205960 |
#' | Decision Tree | 230903 |
#' | Elastic Net | 198443 |

#' We think that an ensemble model will yield better results. Fortunately,
#' scikit-learn provides a very simple, yet effective ensemble classifier,
#' random forest.

rf = RandomForestRegressor()
param_grid = {
    'n_estimators': [1500]
}
grid = GridSearchCV(rf, param_grid, cv=10, n_jobs=-1,
                    scoring='neg_mean_squared_error')
grid.fit(x_train, y_train)
print('Random Forest:')
print('Best params: {}'.format(grid.best_params_))
print('RMSE: {}'.format(math.sqrt(abs(grid.best_score_))))

#' We see that the RMSE for Random Forest is 180122, better than the rest.
#' Because of its effectiveness, let's use this estimator for the final
#' prediction.

prediction = grid.predict(x_test)
mse = mean_squared_error(y_test, prediction)
print('Final Out-of-sample RMSE: {}'.format(math.sqrt(abs(mse))))

#' # Conclusion

#' As we anticipated, an ensemble model, in our case, Random Forest, produced
#' better results than any individual model with an RMSE value of 150472,
#' though only incrementally better given the set of parameters we used for this
#' project. We chose to use 1000 estimators, a low value by any standard, to
#' minimize the training time. The final CV evaluation table is provided below:

#' | Model | RMSE |
#' |---|---|
#' | Ridge Regression | 198433 |
#' | Support Vector Regression | 526903 |
#' | Lasso | 205960 |
#' | Decision Tree | 230903 |
#' | Elastic Net | 198443 |
#' | Random Forest | 150757 |

#' We have learned that a proper set of parameters can have a significant
#' impact on the overall quality of the prediction results, regardless of
#' model is being used. One effective way of searching for the optimal
#' combination of parameters is to use a Grid Search through Cross Validation.
#' This method will iterate through the cartesian product of combinations and
#' cross validate them n-fold times. Although effective and a widely used
#' strategy, one can clearly see how this may increase training time
#' significantly. Thus one must be careful when balancing time vs accuracy.

#' A possible alternative to grid search is to use a randomized parameter
#' search. The idea is similar to grid search but the parameters are chosen
#' randomly from a distribution. Scikit-learn documentation even states that
#' randomized parameter selection could have more favorable
#' properties[^randsearch]

#' Overall, the training process in this project took roughly 30 minutes on my
#' 2012 Macbook Air running a Linux virtual machine through Vagrant. My machine
#' quickly became mostly unusable after completing all training processes. We
#' are certain that with more time or more computing resources, we would be able
#' to produce better results.

#' [^randsearch]: Scikit-learn Documentation (http://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization)

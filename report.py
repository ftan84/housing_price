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

#+ echo=False
import numpy as np
import pandas as pd
import pprint
import pweave
import sqlite3

#' # Training Data
#' The most important element of any data science project is the data itself.
#' This project heavily utilizes data from Zillow, a real estate destination
#' for the internet generation. Fortunately, Zillow provides a public API
#' which provides a convenience to an otherwise tedious task. Below are some
#' basic information of the data.

#+ echo=False
conn = sqlite3.connect('zillowdata.db')
q = '''SELECT *
       FROM properties
       WHERE lastSoldPrice IS NOT NULL'''
training_raw = pd.read_sql_query(q, conn)
conn.close()

#+ echo=False, results='tex', caption='Sample of dataset'
print(training_raw.head().to_latex(columns=[
    'street',
    'city',
    'state',
    'zip',
    'lastSoldPrice'
]))

#' Printing the _shape_ attribute shows that we have 2826 observations and 23
#' columns.

#+ term=True
training_raw.shape

#' Finally, printing the _columns_ attribute produces a list of all column
#' names.

#+ term=True
training_raw.columns

#' Since the goal of this project is to predict the sale price, it is obvious
#' that the _lastSoldPrice_ should be the response variable while the other
#' columns can act as feature variables. Of course, some processing such as
#' dummy variable conversion is required before training begins.
#' 
#' ## Data Collection Process
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
#' ## Data Processing
#' 
#' The next step is to process or clean the data. We can immediately see
#' that we need to convert many of these factor variables into dummy
#' variables. This is easily achieved in Pandas using the _get\_dummies()_
#' function.

#+ term=True
training = training_raw
training['zip'] = training['zip'].astype('category')
training['FIPScounty'] = training['FIPScounty'].astype('category')
training['useCode'] = training['useCode'].astype('category')
training['region'] = training['region'].astype('category')
dummies = pd.get_dummies(training, drop_first=True)
dummies.head()

from bs4 import BeautifulSoup
import requests
import sqlite3


def db_init(dbfile):
    """Initializes SQLite database.

    This function creates required tables.

    Args:
        file (str): Name of SQLite database.

    Returns:
        bool: True if successful, False otherwise.
    """
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    q = '''CREATE TABLE IF NOT EXISTS properties (
           zpid integer PRIMARY KEY,
           street text,
           city text,
           state text,
           zip text,
           FIPScounty text,
           useCode text,
           taxAssessmentYear integer,
           taxAssessment real,
           yearBuilt integer,
           lotSizeSqFt integer,
           finishedSqFt integer,
           bathrooms real,
           bedrooms integer,
           lastSoldDate text,
           lastSoldPrice integer,
           zestimate integer,
           zestimateLastUpdated text,
           zestimateValueChange integer,
           zestimateValueLow integer,
           zestimateValueHigh integer,
           zestimatePercentile integer,
           region text,
           UNIQUE(zpid)
           );'''
    c.execute(q)
    conn.commit()
    conn.close()


def get_properties(apikey, seed, host=None, database=None, user=None,
                       password=None):
    """Get list of Zillow Property IDs.

    Gets a list of Zillow Property IDs (zpid) given an API key, otherwise
    known as a Zillow Web Services ID (ZWSID), and loads it into the database
    provided by host, database, user, and password arguments.

    You must register with Zillow at www.zillow.com/webservice/Registration.htm
    to obtain the ZWSID.

    Args:
        apikey (str): Zillow Web Services ID (ZWSID).
        host (str): Database host.
        database (str): Database name.
        user (str): Database user.
        password (str): Password.

    Returns:
        str: None if successful, otherwise returns error message.
    """
    # r = BeautifulSoup(
    #     requests.get('http://www.zillow.com/webservice/GetDeepComps.htm',
    #                  params=params).content,
    #     'xml'
    # )
    # Start with seed
    baseurl = 'http://www.zillow.com/webservice/'
    params = {'zws-id': apikey,
              'address': seed['address'],
              'citystatezip': seed['zip']}
    r = BeautifulSoup(
            requests.get(baseurl + 'GetDeepSearchResults.htm',
                         params=params).content,
            'xml'
    )
    address = r.response.results.result.address
    q = '''
        INSERT INTO properties (
            zpid,
            street,
            city,
            state,
            zip,
            FIPScounty,
            useCode,
            taxAssessmentYear,
            taxAssessment,
            yearBuilt,
            lotSizeSqFt,
            finishedSqFt,
            bathrooms,
            bedrooms,
            lastSoldDate,
            lastSoldPrice,
            zestimate,
            zestimateLastUpdated,
            zestimateValueChange,
            zestimateValueLow,
            zestimateValueHigh,
            zestimatePercentile,
            region
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?)'''

    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    c.execute(q, (r.response.results.result.zpid.text,
                  address.street.text,
                  address.city.text,
                  address.state.text,
                  address.zipcode.text,
                  r.FIPScounty.text,
                  r.useCode.text,
                  r.taxAssessmentYear.text,
                  r.taxAssessment.text,
                  r.yearBuilt.text,
                  r.lotSizeSqFt.text,
                  r.finishedSqFt.text,
                  r.bathrooms.text,
                  r.bedrooms.text,
                  r.lastSoldDate.text,
                  r.lastSoldPrice.text,
                  r.zestimate.amount.text,
                  r.zestimate.find('last-updated').text,
                  r.zestimate.valueChange.text,
                  r.zestimate.valuationRange.low.text,
                  r.zestimate.valuationRange.high.text,
                  r.zestimate.percentile.text,
                  r.region.attrs['name']))
    conn.commit()
    conn.close()

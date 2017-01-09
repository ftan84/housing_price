from bs4 import BeautifulSoup
import requests
import sqlite3
import time


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


def get_properties(apikey, seed=None, host, database, user=None,
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
    # Start with seed --------------------------------------------------------
    if seed != None:
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
            INSERT OR IGNORE INTO properties (
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
        c.execute(q, (
            '' if r.response.results.result.zpid is None
                else r.response.results.result.zpid.text,
            '' if address.street is None else address.street.text,
            '' if address.city is None else address.city.text,
            '' if address.state is None else address.state.text,
            '' if address.zipcode is None else address.zipcode.text,
            '' if r.FIPScounty is None else r.FIPScounty.text,
            '' if r.useCode is None else r.useCode.text,
            '' if r.taxAssessmentYear is None else r.taxAssessmentYear.text,
            '' if r.taxAssessment is None else r.taxAssessment.text,
            '' if r.yearBuilt is None else r.yearBuilt.text,
            '' if r.lotSizeSqFt is None else r.lotSizeSqFt.text,
            '' if r.finishedSqFt is None else r.finishedSqFt.text,
            '' if r.bathrooms is None else r.bathrooms.text,
            '' if r.bedrooms is None else r.bedrooms.text,
            '' if r.lastSoldDate is None else r.lastSoldDate.text,
            '' if r.lastSoldPrice is None else r.lastSoldPrice.text,
            '' if r.zestimate.amount.text is None else r.zestimate.amount.text,
            '' if r.zestimate.find('last-updated') is None
                else r.zestimate.find('last-updated').text,
            '' if r.zestimate.valueChange is None
                else r.zestimate.valueChange.text,
            '' if r.zestimate.valuationRange.low is None
                else r.zestimate.valuationRange.low.text,
            '' if r.zestimate.valuationRange.high is None
                else r.zestimate.valuationRange.high.text,
            '' if r.zestimate.percentile is None
                else r.zestimate.percentile.text,
            '' if r.region is None else r.region.attrs['name']
        ))
        conn.commit()
        conn.close()
    # Loop through zpid and populate comps until exhausted  -------------------
    # TODO: Keep a list of zpid whose comps we've already requested
    checked_zpids = []
    for i in range(10):
        print('Loop # {}'.format(i))
        conn = sqlite3.connect(dbfile)
        c = conn.cursor()
        results = c.execute('SELECT zpid FROM properties')
        zpids = []
        for row in results:
            zpids.append(row[0])
        conn.commit()
        conn.close()

        for zpid in zpids:
            print('Searching for zpid {} comps...'.format(zpid))
            conn = sqlite3.connect(dbfile)
            c = conn.cursor()
            results = c.execute('''SELECT street FROM properties
                                 WHERE zpid = ?''',
                                 (unicode(str(zpid), 'utf-8'), ))
            address = results.fetchone()[0]
            conn.commit()
            conn.close()
            # if address is not None and i > 0:
            #     print('{} already populated. Skipping'.format(zpid))
            #     continue
            if zpid in checked_zpids:
                print('{} already checked for comps. Skipping'.format(zpid))
                continue
            checked_zpids.append(zpid)
            params = {'zws-id': apikey,
                      'zpid': zpid,
                      'count': 25}
            r = BeautifulSoup(
                requests.get(baseurl + 'GetDeepComps.htm',
                             params=params).content,
                'xml'
            )
            if (r.find('limit-warning') is not None and 
                    r.find('limit-warning').text == 'true'):
                print('Reached api limit. Waiting 24 hours')
                time.sleep(24 * 60 * 60)  # Reached limit, wait 24 hours
            if r.code.text != '0':
                continue
            comp_details = []
            for comp in r.response.properties.find_all('comp'):
                comp_details.append(
                    ('' if comp.zpid is None else comp.zpid.text,
                     '' if comp.address.street is None
                        else comp.address.street.text,
                     '' if comp.address.city is None
                        else comp.address.city.text,
                     '' if comp.address.state is None
                        else comp.address.state.text,
                     '' if comp.address.zipcode is None
                        else comp.address.zipcode.text)
                )
            conn = sqlite3.connect(dbfile)
            c = conn.cursor()
            q = '''INSERT OR IGNORE INTO properties (
                       zpid,
                       street,
                       city,
                       state,
                       zip
                   )
                   VALUES (?, ?, ?, ?, ?);'''
            c.executemany(q, comp_details)
            conn.commit()
            conn.close()
            if (r.find('limit-warning') is not None and
                    r.find('limit-warning') == 'true'):
                break


def get_properties(apikey, host, database, user=None,
                       password=None):
    baseurl = 'http://www.zillow.com/webservice/'
    # Populate property details -----------------------------------------------
    q = '''
        UPDATE properties
        SET street = ?,
            city = ?,
            state = ?,
            zip = ?,
            FIPScounty = ?,
            useCode = ?,
            taxAssessmentYear = ?,
            taxAssessment = ?,
            yearBuilt = ?,
            lotSizeSqFt = ?,
            finishedSqFt = ?,
            bathrooms = ?,
            bedrooms = ?,
            lastSoldDate = ?,
            lastSoldPrice = ?,
            zestimate = ?,
            zestimateLastUpdated = ?,
            zestimateValueChange = ?,
            zestimateValueLow = ?,
            zestimateValueHigh = ?,
            zestimatePercentile = ?,
            region = ?
        WHERE zpid = ?'''
    conn = sqlite3.connect(dbfile)
    c = conn.cursor()
    results = c.execute('SELECT street, zip from properties;')
    rows = results.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        params = {'zws-id': apikey,
                  'address': row[0],
                  'citystatezip': row[1]}
        r = BeautifulSoup(
                requests.get(baseurl + 'GetDeepSearchResults.htm',
                             params=params).content,
                'xml'
        )
        if (r.find('limit-warning') is not None and
                r.find('limit-warning').text == 'true'):
            print('Reached api limit. Waiting 24 hours')
            time.sleep(24 * 60 * 60)  # Reached limit, wait 24 hours
        if r.code.text != '0':
            continue
        address = r.response.results.result.address
        print('Populating housing details for zpid {}'.format(
            r.response.results.result.zpid.text
        ))
        conn = sqlite3.connect(dbfile)
        c = conn.cursor()
        c.execute(q, (
            '' if address.street is None else address.street.text,
            '' if address.city is None else address.city.text,
            '' if address.state is None else address.state.text,
            '' if address.zipcode is None else address.zipcode.text,
            '' if r.FIPScounty is None else r.FIPScounty.text,
            '' if r.useCode is None else r.useCode.text,
            '' if r.taxAssessmentYear is None else r.taxAssessmentYear.text,
            '' if r.taxAssessment is None else r.taxAssessment.text,
            '' if r.yearBuilt is None else r.yearBuilt.text,
            '' if r.lotSizeSqFt is None else r.lotSizeSqFt.text,
            '' if r.finishedSqFt is None else r.finishedSqFt.text,
            '' if r.bathrooms is None else r.bathrooms.text,
            '' if r.bedrooms is None else r.bedrooms.text,
            '' if r.lastSoldDate is None else r.lastSoldDate.text,
            '' if r.lastSoldPrice is None else r.lastSoldPrice.text,
            '' if r.zestimate.amount is None else r.zestimate.amount.text,
            '' if r.zestimate.find('last-updated') is None
                else r.zestimate.find('last-updated').text,
            '' if r.zestimate.valueChange is None
                else r.zestimate.valueChange.text,
            '' if r.zestimate.valuationRange.low is None
                else r.zestimate.valuationRange.low.text,
            '' if r.zestimate.valuationRange.high is None
                else r.zestimate.valuationRange.high.text,
            '' if r.zestimate.percentile is None
                else r.zestimate.percentile.text,
            '' if r.region is None else r.region.attrs['name'],
            '' if r.response is None else r.response.results.result.zpid.text
        ))
        conn.commit()
        conn.close()

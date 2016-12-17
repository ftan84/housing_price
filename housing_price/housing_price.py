from bs4 import BeautifulSoup
import requests

def collect_properties(apikey, host=None, database=None, user=None,
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
    params = {'zws-id': apikey,
              'address': '2114 Bigelow Ave',
              'citystatezip': 'Seattle, WA'}
    r = BeautifulSoup(
        requests.get('http://www.zillow.com/webservice/GetSearchResults.htm',
                     params=params).content,
        'xml'
    )





    params = {'zws-id': token,
              'zpid': '48749425',
              'count': '3'}
    r = BeautifulSoup(
        requests.get('http://www.zillow.com/webservice/GetDeepComps.htm',
                     params=params).content,
        'xml'
    )

from flask import Flask, jsonify, request
import sqlite3
import yaml


app = Flask(__name__)
with open('config.yml', 'r') as f:
    config = yaml.load(f)


@app.route('/api/housing-predictor/v1/getProperties')
def getProperties():
    city = str(request.args.get('city'))
    useCode = str(request.args.get('useCode'))
    bedrooms = str(request.args.get('bedrooms'))
    bathrooms = str(request.args.get('bathrooms'))
    limit = str(request.args.get('limit'))
    con = sqlite3.connect(config['dbfile'])
    c = con.cursor()
    q = '''SELECT *
           FROM properties
           WHERE lastSoldPrice NOT NULL
           AND city LIKE ?
           AND useCode LIKE ?
           AND bedrooms LIKE ?
           AND bathrooms LIKE ?
           ORDER BY RANDOM()
           LIMIT ?'''
    c.execute(q, (city, useCode, bedrooms, bathrooms, limit))
    r = [dict((c.description[i][0], value) \
            for i, value in enumerate(row)) for row in c.fetchall()]
    response = jsonify(r)
    response.headers.add('Access-Control-Allow-Origin', '*')
    con.commit()
    con.close()
    return response

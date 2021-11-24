from flask import Flask
from flask import request
import json

app = Flask(__name__)

energy_today = ''
energy_tomorrow = ''

water_today = ''
water_tomorrow = ''

security_label = ''
security_status = ''

warnings = ['', '', '', '', '', '', '']

voltage = '-V'
current = '-mA'
watts = '-W'


@app.route('/', methods=['GET'])
def home():
    return 'add to url /energy, /water, /security or /warning'


@app.route('/energy', methods=['GET'])
def energy():
    data_set = {'today': energy_today, 'tomorrow': energy_tomorrow}
    dump = json.dumps(data_set)

    return dump


@app.route('/post_energy', methods=['GET', 'POST'])
def post_energy():
    global energy_today, energy_tomorrow

    energy_today = str(request.args.get('today'))
    energy_tomorrow = str(request.args.get('tomorrow'))

    return json.dumps({'status': 'ok'})


@app.route('/water', methods=['GET'])
def water():
    data_set = {'today': water_today, 'tomorrow': water_tomorrow}
    dump = json.dumps(data_set)

    return dump


@app.route('/post_water', methods=['GET', 'POST'])
def post_water():
    global water_today, water_tomorrow

    water_today = str(request.args.get('today'))
    water_tomorrow = str(request.args.get('tomorrow'))

    return json.dumps({'status': 'ok'})


@app.route('/security', methods=['GET'])
def security():
    data_set = {'label': security_label, 'status': security_status}
    dump = json.dumps(data_set)

    return dump


@app.route('/post_security', methods=['GET', 'POST'])
def post_security():
    global security_label, security_status

    security_label = str(request.args.get('label'))
    security_status = str(request.args.get('status'))

    return json.dumps({'status': 'ok'})


@app.route('/warning', methods=['GET'])
def warning():
    data_set = {'warning0': warnings[6], 'warning1': warnings[5], 'warning2': warnings[4],
                'warning3': warnings[3], 'warning4': warnings[2], 'warning5': warnings[1],
                'warning6': warnings[0]}
    dump = json.dumps(data_set)

    return dump


@app.route('/post_warning', methods=['GET', 'POST'])
def post_warning():
    global warnings

    warnings.append(str(request.args.get('warning')))
    del warnings[0]

    return json.dumps({'status': 'ok'})


@app.route('/live', methods=['GET'])
def live():
    if voltage == '-V' and current == '-mA':
        data_set = {'voltage': '-V', 'current': '-mA', 'watts': '-W'}
    else:
        watts = voltage * current / 1000
        data_set = {'voltage': f'{voltage:.4f}V', 'current': f'{current:.4f}mA', 'watts': f'{watts:.4f}W'}
    dump = json.dumps(data_set)

    return dump


@app.route('/post_live', methods=['GET', 'POST'])
def post_live():
    global voltage, current

    voltage = float(request.args.get('voltage'))
    current = float(request.args.get('current'))

    return json.dumps({'status': 'ok'})

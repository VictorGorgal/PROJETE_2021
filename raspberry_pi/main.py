from energy_model import energy_pred
from water_model import water_pred
from datetime import datetime
from requests import get
import numpy as np
import pyrebase


class centralHub:
    def __init__(self):
        self.WARNING_DICT = {'energy': 'energia', 'water': 'água'}
        self.URL = 'http://projetemt3015e.pythonanywhere.com/post_'  # base url

        self.energy_model = energy_pred()  # loads models
        self.water_model = water_pred()

        firebase = pyrebase.initialize_app({'apiKey': 'AIzaSyBflU5Lbj0s0WIQ7GUR-EuEk5Ga7Dap4Gs',
                                            'authDomain': 'teste-8b116.firebaseapp.com',
                                            'databaseURL': 'https://teste-8b116.firebaseio.com/',
                                            'storageBucket': 'teste-8b116.appspot.com'})
        self.database = firebase.database()
        self.stream = self.database.child().stream(self.stream_handler)

        self.energy_counter = 6
        self.last_energy_out = None
        self.last_water_out = None

        self.AA = False

        date = datetime.now()
        self.day = date.day
        self.month = date.month
        self.year = date.year

        print('Ready!')

    def main(self, data):  # runs when stream_handler() detects a change on the firebase database
        if data == '1':  # generates the energy predictions and sends to the API
            print(f'sending energy data: day {self.energy_counter}')

            y, out = self.energy_model.generate_prediction(self.energy_counter)

            self.warning_check(y, self.last_energy_out, 'energy')
            self.last_energy_out = out

            self.energy_counter += 1
            if self.energy_counter == 8:
                self.energy_counter = 6
                self.last_energy_out = None

            y, out = self.to_send_pred(y), self.to_send_pred(out)

            params = {'today': y, 'tomorrow': out}
            get(self.URL + 'energy', params=params)

        elif data == '2':  # generates the water predictions and sends to the API
            print('sending water data')

            y, out = self.water_model.generate_prediction()

            self.warning_check(y, self.last_water_out, 'water')
            self.last_water_out = out

            y, out = self.to_send_pred(y), self.to_send_pred(out)

            params = {'today': y, 'tomorrow': out}
            get(self.URL + 'water', params=params)

        elif data == '3':
            print('Sending security data')

            p1, p2 = self.to_send_warning(('Porta da frente', 0),
                                          ('Porta de tras', 1),
                                          ('Porta da garagem', 1),
                                          ('Janela do quarto', 0),
                                          ('Janela da cozinha', 1))

            params = {'label': p1, 'status': p2}
            get(self.URL + 'security', params=params)

        elif data == '4':
            if not self.AA:
                print('Sending warning water data')

                warning = f'{self.day - 1}/{self.month}/{self.year} | Aviso: Água\n' \
                          f'O seu consumo de água está muito maior do que o previsto!\n' \
                          f'Verifique se não deixou nada ligado/aberto acidentalmente.\n'
                params = {'warning': warning}

                get(self.URL + 'warning', params=params)
            else:
                print('Sending warning security data')

                warning = f'{self.day - 1}/{self.month}/{self.year} | Aviso: Segurança\n' \
                          f'A Porta da frente está aberta por 10 minutos!\n' \
                          f'Verifique se não deixou ela aberta acidentalmente.\n'
                params = {'warning': warning}

                get(self.URL + 'warning', params=params)

            self.AA = not self.AA

        else:  # reads the live data
            data = data.split(' ')

            voltage = float(data[0].replace('V', ''))
            current = float(data[1].replace('mA', ''))

            lamp_data = data[2].split(';')
            lamp1 = int(lamp_data[0])
            lamp2 = int(lamp_data[1])
            lamp3 = int(lamp_data[2])

            params = {'voltage': voltage, 'current': current}
            get(self.URL + 'live', params=params)

            p1, p2 = self.to_send_warning(('Lâmpada do quarto esquerdo', lamp3),
                                          ('Lâmpada do quarto central', lamp1),
                                          ('Lâmpada do quarto direito', lamp2))

            params = {'label': p1, 'status': p2}
            get(self.URL + 'security', params=params)

    def stream_handler(self, message):  # runs main func when firebase is updated
        data = message['data']
        if type(data) == dict:
            data = data['data']
        if data != '0':
            print(data)
            self.database.child().update({'data': '0'})
            try:
                self.main(data)
            except Exception as e:
                print(e)

    def warning_check(self, y, last_out, label):
        if last_out is not None and True in list(y > last_out * 1.8):
            if list(y > last_out * 1.8).count(True) > 5:
                print('Warning!')
                warning = f'{self.day}/{self.month}/{self.year} | Aviso: {self.WARNING_DICT[label].capitalize()}\n' \
                          f'O seu consumo de {self.WARNING_DICT[label]} está muito maior do que o previsto!\n' \
                          f'Verifique se não deixou nada ligado/aberto acidentalmente.\n'
                params = {'warning': warning}
                get(self.URL + 'warning', params=params)

                self.day += 1

    @staticmethod
    def to_send_warning(*args):  # recieves any amount of tuples
        to_return1 = ''
        to_return2 = ''
        for arg in args:
            to_return1 += f'{arg[0]}\n'
            to_return2 += '✓\n' if arg[1] == 1 else '✕\n'
        return to_return1, to_return2

    @staticmethod
    def to_send_pred(lis):
        return ';'.join(np.char.mod('%.3f', lis))


if __name__ == '__main__':
    hub = centralHub()

from time import sleep, time
from machine import Pin, ADC, UART

# from _thread import start_new_thread

uart = UART(1, 115200)

lamp1 = Pin(13, Pin.IN)
lamp2 = Pin(14, Pin.IN)
lamp3 = Pin(15, Pin.IN)

warning_pin = Pin(18, Pin.IN)
security_pin = Pin(19, Pin.IN)
water_pin = Pin(20, Pin.IN)
energy_pin = Pin(21, Pin.IN)

current_pin = ADC(Pin(26))
voltage_pin = ADC(Pin(27))

debouncing = 0.2

voltage = 0

current = 0
mv = 66
aref = 3.3
outv = 3.3 / 2
error = 10

last_warning = 0
last_security = 0
last_water = 0
last_energy = 0

timer = time()

while True:
    if time() > timer:  # run every other second
        for _ in range(100):  # samples 50 measurements
            voltage += voltage_pin.read_u16() / 65535 * aref * 2
            current += ((current_pin.read_u16() / 65535 * aref - outv) * 1000 / mv - error) * 300 * -1

        voltage /= 100  # average
        current /= 100

        current = current if current > 20 else 0  # prevents the sensor reading noise

        to_send = str(voltage) + 'V ' + str(current) + 'mA ' + str(lamp3.value()) + ';' + str(
            lamp2.value()) + ';' + str(lamp1.value()) + '\n'  # sends to pc
        uart.write(to_send)

        voltage = 0  # resets
        current = 0

        timer = time()

    # will be used during the presentation for a better explanation of what the project does
    warning = warning_pin.value()  # reads all buttons
    security = security_pin.value()
    water = water_pin.value()
    energy = energy_pin.value()

    if energy == 1 and last_energy == 0:  # sends a value acording to each button pressed
        uart.write('1\n')
        sleep(debouncing)
    elif water == 1 and last_water == 0:
        uart.write('2\n')
        sleep(debouncing)
    elif security == 1 and last_security == 0:
        uart.write('3\n')
        sleep(debouncing)
    elif warning == 1 and last_warning == 0:
        uart.write('4\n')
        sleep(debouncing)

    last_warning = warning  # used to detect the rising edge
    last_security = security
    last_water = water
    last_energy = energy

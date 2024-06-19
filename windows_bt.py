import serial
import time
from test import get_result

'''
Connect the Bluetooth device to the PC.
Open the serial port software (done via the script).
Configure the serial port with the specified settings.
Set the sending mode to character mode.
Send the specified commands.
'''

# Replace 'COMx' with the actual COM port your Bluetooth device is connected to
COM_PORT = 'COMx' #diff way of using MAC address
BAUD_RATE = 115200
TIMEOUT = 1

# Initialize serial connection
ser = serial.Serial(
    port=COM_PORT,
    baudrate=BAUD_RATE,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=TIMEOUT
)

# Ensure the serial connection is open
if ser.is_open:
    print(f"Connected to {COM_PORT} at {BAUD_RATE} baud.")
else:
    ser.open()
    print(f"Opened connection to {COM_PORT} at {BAUD_RATE} baud.")

# Send commands
commands = ["+AAAAAAAE", "+BBBBBBBBE", "+CCCCCCCE"]
# ser.write("+AAAAAAAE".encode()) # open
while True:
    result = get_result()
    if result not in (0 or 1):
        print("Error occurred")
        break
    elif result == 0:
        ser.write("+AAAAAAAE".encode())
        print("Phone detected")
# ser.write("+CCCCCCCE".encode()) # close

# Close the serial connection
ser.close()
print("Closed serial connection.")
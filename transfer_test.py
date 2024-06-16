import bluetooth
# use pip install git+https://github.com/airgproducts/pybluez2.git@0.46 to install 
# Discover nearby devices
nearby_devices = bluetooth.discover_devices(lookup_names=True)
print("Found {} devices".format(len(nearby_devices)))

# for addr, name in nearby_devices:
#     print("  {} - {}".format(addr, name))


# Replace 'XX:XX:XX:XX:XX:XX' with your iPhone's Bluetooth address
target_address = '4B:FB:A7:15:64:32'
'''
# Establish a connection to the target device
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((target_address, 1))

# Send data (0 or 1)
status = 0  # or 1
sock.send(str(status))

sock.close()
'''
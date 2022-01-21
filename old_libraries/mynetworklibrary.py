import tinytuya
import netifaces
from config import *

def find_tuya():
    TuyaDevices = tinytuya.deviceScan(False, 5)
    string_of_devices = list(TuyaDevices.keys())
    string_of_devices = "".join(string_of_devices)
    return string_of_devices

def get_local_ip():
    for ifaceName in netifaces.interfaces():
        if ifaceName == "wlan0":
            addrs = netifaces.ifaddresses(ifaceName)
            out = addrs[netifaces.AF_INET][0]['addr']
            return out

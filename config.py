import glob

LCD_RS = 7 # Pi pin 26
LCD_E = 8 # Pi pin 24
LCD_D4 = 25 # Pi pin 22
LCD_D5 = 24 # Pi pin 18
LCD_D6 = 23 # Pi pin 16
LCD_D7 = 18 # Pi pin 12

# Device constants
LCD_CHR = True # Character mode
LCD_CMD = False # Command mode
LCD_CHARS = 16 # Characters per line (16 max)
LCD_LINE_1 = 0x80 # LCD memory location for 1st line
LCD_LINE_2 = 0xC0 # LCD memory location 2nd line
BUZZER = 22

#Tuya GWID:
TUYA_GWID = "bf589a7166e99efc42sod9"
TUYA_KEYID = "a8d79cd6990a3b75"
SousVide_ip = "10.0.0.20"

base_dir = '/sys/bus/w1/devices/'
device_folder = glob.glob(base_dir + '28*')[0]
device_file = device_folder + '/w1_slave'

#W1ThermSensor
SENSORADDRESS = "03159779e855"


#Phase cycle in seconds
PHASE_LENGTH = 1
WAIT_PERIOD = 0.01
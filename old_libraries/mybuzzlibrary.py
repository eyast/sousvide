import RPi.GPIO as GPIO
import time
from config import *


def buzz(duration, times):
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(BUZZER,GPIO.OUT)
    if times > 0:
        for _ in range(times):
            GPIO.output(BUZZER, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(BUZZER, GPIO.LOW)
            time.sleep(duration)
    else:
        GPIO.output(BUZZER, GPIO.HIGH)

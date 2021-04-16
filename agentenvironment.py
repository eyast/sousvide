
import tinytuya
from mynetworklibrary import get_local_ip, find_tuya
from mydisplaylibrary import *
from mytemperaturelibrary import *
import datetime
from config import *
import copy
import os

class Environment():
    def __init__(self, target_temp, target_duration, target_step_size):
        self.stepcount = 0
        self.target_temp = target_temp
        self.target_duration = target_duration
        self.target_step_size = target_step_size
        self.is_over = False
        initialize_lcd()
        self.emergency_off = False
        self.Raspberry_ip = self.get_local_ip()
        self.SousVide_ip = self.get_sous_vide_ip()
        self.TUYA_GWID = TUYA_GWID
        self.TUYA_KEYID = TUYA_KEYID
        self.tuya_properties = self.generate_tuya_properties()
        self.switch_status = self.get_tuya_status()
        self.reset_tuya_switch()
        assert not self.switch_status
        self.temperature_sensor_works = self.check_temperature_sensor_works()
        self.temperature = self.get_temperature()
        self.time_start = self.get_local_time()
        self.verbose_log_file_path = self.generate_file_name()
        self.previous = None

    def get_local_ip(self):
        local_ip = get_local_ip()
        lcd_text(local_ip,LCD_LINE_1)
        return local_ip

    def get_sous_vide_ip(self):
        tuya_ip = find_tuya()
        if tuya_ip == '':
            lcd_text("Cant find Tuya",LCD_LINE_2)
            self.emergency_off = True
            self.shutdown()
        assert tuya_ip != ''
        lcd_text(tuya_ip,LCD_LINE_2)
        return tuya_ip

    def generate_tuya_properties(self):
        my_tuya = tinytuya.OutletDevice(self.TUYA_GWID, self.SousVide_ip, self.TUYA_KEYID)
        my_tuya.set_version(3.3)
        return my_tuya

    def reset_tuya_switch(self):
        self.tuya_off()

    def get_tuya_status(self):
        self.tuya_properties = self.generate_tuya_properties()
        data = self.tuya_properties.status()
        data = data['dps']['1']
        return data

    def check_temperature_sensor_works(self):
        tempfile_exists = file_exists()
        if not tempfile_exists:
            lcd_text("No temp sensor",LCD_LINE_2)
            self.emergency_off = True
            self.shutdown()
        assert tempfile_exists
        return tempfile_exists
        
    def get_temperature(self):
        if self.check_temperature_sensor_works():
            temp = read_temp()
            self.temperature = temp
            return temp

    def get_local_time(self):
        now = datetime.datetime.now()
        now = str(now)
        return now

    def shutdown(self):
        # make a buzz
        # Force turn off the Tuya
        pass

    def generate_file_name(self):
        run_name = str(self.time_start)
        run_name = run_name + ".csv"
        return run_name

    def get_actions(self):
        return [True, False]

    def tuya_off(self):
        if self.switch_status:
            self.tuya_properties.set_status(False, 1)
            self.switch_status = False

    def tuya_on(self):
        if not self.switch_status:
            self.tuya_properties.set_status(True, 1)
            self.switch_status = True

    def get_observation(self):
        current_status = self.get_tuya_status()
        possible_actions = self.get_actions()
        current_temperature = self.get_temperature()
        if current_temperature > self.target_temp:
            current_reward = -100
        else:
            current_reward = self.target_temp - round(current_temperature)  
        return [current_status, possible_actions, current_temperature, current_reward]

    def apply_step(self, step):
        #self.previous = copy.deepcopy(self)
        
        self.stepcount += 1
        if step == True:
            self.tuya_on()
        elif step == False:
            self.tuya_off()
        temp = self.get_temperature()
        temp = str(temp)
        lcd_text(temp, LCD_LINE_1)
        return self.target_step_size

class Agent():
    def __init__(self, Environment):
        self.Environment = Environment
        self.reward = 0


    def take_step(self):
        current_reward = self.Environment.get_observation()[3]
        self.reward = self.reward + current_reward
        if self.Environment.temperature < self.Environment.target_temp:            
            waitperiod = self.Environment.apply_step(True)
        else:
            waitperiod = self.Environment.apply_step(False)
        time.sleep(waitperiod)
        what_to_display = str(self.Environment.stepcount) #+"-" + str(self.reward)
        lcd_text(what_to_display, LCD_LINE_2)
        self.log_to_file()

    def log_to_file(self):
        filename = self.Environment.verbose_log_file_path
        stepcount = self.Environment.stepcount
        actiontime = self.Environment.get_local_time()
        target_temp = self.Environment.target_temp
        target_duration = self.Environment.target_duration
        current_temp = self.Environment.temperature
        current_reward = self.reward
        switch_status = self.Environment.switch_status
        dir = "logs/"
        logfile = dir + filename
        headerdata = ["stepcount", "actiontime", "target_temp", "target_duration" , "current_temp" , "current_reward" , "switch_status" ]
        logdata = [stepcount, actiontime, target_temp, target_duration, current_temp, current_reward, switch_status]
        logdata = [str(x) for x in logdata]
        headerdata = ",".join(headerdata)
        logdata = ",".join(logdata)
        if os.path.exists(logfile):
            f = open(logfile, 'a')
            f.write(logdata)
            f.write("\n")
            f.close()
        else:
            f = open(logfile, 'w')
            f.write(headerdata)
            f.write("\n")
            f.write(logdata)
            f.write("\n")
            f.close()
        print(logdata)

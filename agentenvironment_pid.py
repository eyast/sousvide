from datetime import datetime, timedelta
import os
import time
import logging
import tinytuya
from mynetworklibrary import get_local_ip, find_tuya
from mydisplaylibrary import *
from mytemperaturelibrary import *
from mybuzzlibrary import *
from config import *


class Environment():
    def __init__(self, phase_cycle_in_sec):
        self._logger = logging.getLogger(type(self).__name__)
        self.stepcount = 0
        self.phase_cycle_in_sec = phase_cycle_in_sec
        self.is_over = False
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
        self.possible_actions = self.get_actions()

    def get_local_ip(self):
        local_ip = get_local_ip()
        print(local_ip)
        return local_ip

    def get_sous_vide_ip(self):
        tuya_ip = find_tuya()
        if tuya_ip == "":
            self.emergency_off = True
            self.shutdown()
        assert tuya_ip != ""
        print(tuya_ip)
        return tuya_ip

    def generate_tuya_properties(self):
        my_tuya = tinytuya.OutletDevice(self.TUYA_GWID, self.SousVide_ip, 
                                        self.TUYA_KEYID)
        my_tuya.set_version(3.3)
        return my_tuya

    def reset_tuya_switch(self):
        self.tuya_off()

    def get_tuya_status(self):
        self.tuya_properties = self.generate_tuya_properties()
        data = self.tuya_properties.status()
        data = data["dps"]["1"]
        return data

    def check_temperature_sensor_works(self):
        tempfile_exists = file_exists()
        if not tempfile_exists:
            self.emergency_off = True
            self.shutdown()
        assert tempfile_exists
        print("Temp sensor exists")
        return tempfile_exists
        
    def get_temperature(self):
        temp = read_temp()
        self.temperature = temp
        return temp

    def get_local_time(self):
        now = datetime.now() +  timedelta(hours=9)
        now = now.strftime("%H-%M-%S") 
        return now

    def shutdown(self):
        self.is_over = True
        self.tuya_off()
        
    def generate_file_name(self):
        run_name = str(self.time_start)
        run_name = run_name + ".csv"
        return run_name

    def get_actions(self):
        possibilities = [0, self.phase_cycle_in_sec]
        return possibilities

    def tuya_off(self):
        if self.switch_status:
            self.tuya_properties.set_status(False, 1)
            self.switch_status = False
        else:
            pass

    def tuya_on(self):
        if not self.switch_status:
            self.tuya_properties.set_status(True, 1)
            self.switch_status = True
        else:
            pass

    def apply_step(self, duration):
        assert (duration >= self.possible_actions[0]) or \
                (duration <= self.possible_actions[-1])
        assert not self.is_over
        self.stepcount += 1
        if duration == self.possible_actions[0]:
            self.tuya_off()
            time.sleep(self.phase_cycle_in_sec)
        elif duration == self.possible_actions[-1]:
            self.tuya_on()
            time.sleep(self.phase_cycle_in_sec)
        else:
            time_start = time.time()
            time_end = time.time()
            duration_on = duration
            duration_off = self.possible_actions[-1] - duration_on
            while (time_end - time_start) <= duration_on:
                self.tuya_on()
                time_end = time.time()
            time_start = time.time()
            time_end = time.time()
            while (time_end - time_start) <= duration_off:
                self.tuya_off()
                time_end = time.time()
        self.temperature = self.get_temperature()


class Agent():
    def __init__(self, kP, kI, kD, target_temp, target_duration, Environment, 
                label):
        self._logger = logging.getLogger(type(self).__name__)
        self.Environment = Environment
        self.target_temp = target_temp
        self.target_duration = target_duration
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.Pval = 0
        self.Ival = 0
        self.Dval = 0
        self.MaxIVal =  self.Environment.phase_cycle_in_sec 
        self.MinIVal = 0
        self.error = 0
        self.error_history = []
        self.P_history = []
        self.I_history = []
        self.D_history = []
        self.last_outcome = None
        self.input = None
        self.last_input = None
        self.reached_target_temp = False
        self.reached_target_temp_at_timestamp = None
        self.done = False
        self.label = label

    def update_error(self):
        current_error = self.target_temp - self.input
        self.error = current_error

    def P(self):
        self.update_error()
        self.Pval = self.error
        return self.Pval

    def I(self):
        self.update_error()
        self.Ival = self.Ival + self.error
        if self.Ival > self.MaxIVal:
            self.Ival = self.MaxIVal
        if self.Ival < self.MinIVal:
            self.Ival = self.MinIVal
        return self.Ival

    def D(self):
        if self.last_input is None:
            self.Dval = 0
        else:
            self.Dval = self.input - self.last_input
        return self.Dval

    def ReturnPID(self):
        if self.kP != 0:
            outcome_P = self.kP * self.P()
        else:
            outcome_P = 0
        if self.kI != 0:
            outcome_I = self.kI * self.I()
        else:
            outcome_I = 0
        if self.kD != 0:
            outcome_D = -(self.kD * self.D())
        else:
            outcome_D = 0
        outcome = outcome_P + outcome_I + outcome_D
        outcome = round(outcome, 3)
        if outcome >= self.Environment.possible_actions[-1]:
            outcome = self.Environment.possible_actions[-1]
        if outcome <= self.Environment.possible_actions[0]:
            outcome = self.Environment.possible_actions[0]
        self.last_outcome = outcome
        self._logger.debug(f"P: {outcome_P}")
        self._logger.debug(f"I: {outcome_I}")
        self._logger.debug(f"D: {outcome_D}")
        self._logger.debug(f"output: {self.last_outcome}")
        return outcome

    def take_step(self):
        assert not self.Environment.is_over
        self.input = self.Environment.get_temperature()
        if self.usequeue:
            self.UpdateQ()
        movement = self.ReturnPID()
        self.log_to_file()
        self.Environment.apply_step(movement)
        self.last_input = self.input
        if not self.reached_target_temp:
            if self.input >= self.target_temp:
                self.reached_target_temp = True
                buzz(0.5, 4)
                print(f"Buzzed at: {datetime.now()}")
                self.reached_target_temp_at_timestamp = datetime.now()
        if self.reached_target_temp:
            now = datetime.now()
            elapsed = (now - self.reached_target_temp_at_timestamp).seconds
            if elapsed > self.target_duration * 60:
                if not self.done:
                    self.done = True
                    buzz(1, 6)
                    print(f"finished at: {datetime.now()}")
        

    def log_to_file(self):
        prefix = self.label + \
                f"_KP.{self.kP}_KI.{self.kI}_KD.{self.kD}" + \
                f"_Cycles.{self.Environment.phase_cycle_in_sec}" + \
                f"_TargetTemp.{self.target_temp}_"
        filename = self.Environment.verbose_log_file_path
        dir = "logs/"
        logfile = dir + prefix + filename
        headerdata = ["stepcount", "actiontime", "target_temp" , \
                    "current_temp" , "Pval", "Ival", "Dval", "outcome"]
        logdata = [self.Environment.stepcount, \
                    self.Environment.get_local_time(), \
                    self.target_temp, self.Environment.temperature, \
                    self.Pval, self.Ival, self.Dval, self.last_outcome]
        logdata = [str(x) for x in logdata]
        headerdata = ",".join(headerdata)
        logdata = ",".join(logdata)
        if os.path.exists(logfile):
            f = open(logfile, "a")
            f.write(logdata)
            f.write("\n")
            f.close()
        else:
            f = open(logfile, "w")
            f.write(headerdata)
            f.write("\n")
            f.write(logdata)
            f.write("\n")
            f.close()
        print(logdata.split(","))

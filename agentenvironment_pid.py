from datetime import datetime, timedelta
import os
import time
import logging
import multiprocessing
import sys
import tinytuya
from mynetworklibrary import get_local_ip, find_tuya
from mydisplaylibrary import *
from mybuzzlibrary import *
from w1thermsensor import W1ThermSensor, Sensor
from config import *


class Environment(multiprocessing.Process):
    def __init__(self, phase_cycle_in_sec, TemperatureQueue, StatusQueue, 
                MovementQueue):
        multiprocessing.Process.__init__(self, group=None, 
            name="Environment_Process")
        self._logger = logging.getLogger(type(self).__name__)
        os.system('modprobe w1-gpio')
        os.system('modprobe w1-therm')
        self.phase_cycle_in_sec = phase_cycle_in_sec
        self.is_over = False
        self.SousVide_ip = SousVide_ip
        self.TUYA_GWID = TUYA_GWID
        self.TUYA_KEYID = TUYA_KEYID
        self.tuya_properties = self.generate_tuya_properties()
        self.switch_status = self.get_tuya_status()
        self.reset_tuya_switch()
        assert not self.switch_status
        self.sensor = W1ThermSensor(Sensor.DS18B20, SENSORADDRESS)
        self.sensor.set_resolution(resolution=11, persist=False)
        self.temperature = self.get_temperature()
        self.TemperatureQueue = TemperatureQueue
        self.StatusQueue = StatusQueue
        self.MovementQueue = MovementQueue

    def generate_tuya_properties(self):
        my_tuya = tinytuya.OutletDevice(self.TUYA_GWID, self.SousVide_ip, 
                                        self.TUYA_KEYID)
        my_tuya.set_version(3.3)
        #my_tuya.set_socketPersistent(True)
        return my_tuya

    def reset_tuya_switch(self):
        self.tuya_off()

    def get_tuya_status(self):
        self.tuya_properties = self.generate_tuya_properties()
        data = self.tuya_properties.status()
        data = data["dps"]["1"]
        return data
        
    def get_temperature(self):
        self.temperature = self.sensor.get_temperature()
        return self.temperature

    def shutdown(self):
        self.is_over = True
        self.tuya_off()        

    def tuya_off(self):
        if self.switch_status:
            self.tuya_properties.set_status(False, 1)
            # self.tuya_properties.turn_off(switch=1)
            self.switch_status = False
        else:
            pass

    def tuya_on(self):
        if not self.switch_status:
            #self.tuya_properties.set_status(True, 1)
            self.tuya_properties.turn_on(switch=1)
            self.switch_status = True
        else:
            pass

    def apply_step(self, duration):
        assert (duration >= 0) or \
                (duration <= self.phase_cycle_in_sec)
        assert not self.is_over
        if duration == 0:
            self.tuya_off()
            time.sleep(self.phase_cycle_in_sec)
        elif duration == self.phase_cycle_in_sec:
            self.tuya_on()
            time.sleep(duration)
        else:
            duration_on = duration
            duration_off = self.phase_cycle_in_sec - duration_on     
            self.tuya_on()
            time.sleep(duration_on)
            self.tuya_off()
            time.sleep(duration_off)

    def run(self):
        while not self.is_over:
            self.temperature = self.get_temperature()
            self.TemperatureQueue.put(self.temperature)
            while not self.MovementQueue.empty():
                movement = self.MovementQueue.get()
                self.apply_step(movement)


class Agent(multiprocessing.Process):
    def __init__(self, kP, kI, kD, target_temp, target_duration, 
                TemperatureQueue, StatusQueue, MovementQueue, label, 
                length):
        multiprocessing.Process.__init__(self, group=None,
            name="Agent_Process")
        self._logger = logging.getLogger(type(self).__name__)
        # time.sleep(1)
        self.stepcount = 0
        self.target_temp = target_temp
        self.target_duration = target_duration
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.Pval = 0
        self.Ival = 0
        self.Dval = 0
        self.MaxIVal =  PHASE_LENGTH
        self.MinIVal = 0
        self.error = 0
        self.last_outcome = None
        self.last_input = None
        self.reached_target_temp = False
        self.reached_target_temp_at_timestamp = None
        self.done = False
        self.label = label
        self.length = length
        self.TemperatureQueue = TemperatureQueue
        self.StatusQueue = StatusQueue
        self.MovementQueue = MovementQueue
        # time.sleep(2)
        self.input = self.TemperatureQueue.get()
        self.movement = 0

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
        self.movement = outcome
        outcome = round(outcome, 1)
        if outcome >= PHASE_LENGTH:
            outcome = PHASE_LENGTH
        if outcome <= 0:
            outcome = 0
        self.last_outcome = outcome
        self._logger.debug(f"P: {outcome_P}")
        self._logger.debug(f"I: {outcome_I}")
        self._logger.debug(f"D: {outcome_D}")
        self._logger.debug(f"output: {self.last_outcome}")
        return outcome

    def run(self):
        # Add a while loop to make sure it does this
        # as long as there is no "empty" message in the Queue
        
        while not self.done:
            self.input = self.TemperatureQueue.get()
            movement = self.ReturnPID()   
            self.MovementQueue.put(movement)
            self.log_to_file()
            self.last_input = self.input
            self.stepcount += 1
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
        else:
            time.sleep(WAIT_PERIOD)
            print("Queue empty, waiting")
            # if self.length < 0:
            #     pass
            # elif self.Environment.stepcount >= self.length:
            #     sys.exit()

    def get_local_time(self):
        now = datetime.now() +  timedelta(hours=9)
        now = now.strftime("%H-%M-%S") 
        return now

    def log_to_file(self):
        prefix = self.label + \
                f"_KP.{self.kP}_KI.{self.kI}_KD.{self.kD}" + \
                f"_Cycles.{PHASE_LENGTH}" + \
                f"_TargetTemp.{self.target_temp}_"
        dir = "logs/"
        logfile = dir + prefix + ".csv"
        headerdata = ["stepcount", "actiontime", "target_temp" , \
                    "current_temp" , "Pval", "Ival", "Dval", "outcome", 
                    "movement"]
        logdata = [self.stepcount, \
                    self.get_local_time(), \
                    self.target_temp, self.input, \
                    self.Pval, self.Ival, self.Dval, self.last_outcome,
                    self.movement]
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
        # print(logdata.split(","))

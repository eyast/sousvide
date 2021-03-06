import multiprocessing as mp
import logging
from agentenvironment_pid import RiceCookerController, Agent, TemperatureProvider
from config import PHASE_LENGTH



# https://www.seriouseats.com/2013/10/sous-vide-101-all-about-eggs.html
# https://www.wescottdesign.com/articles/pid/pidWithoutAPhd.pdf
# http://brettbeauregard.com/blog/2011/04/improving-the-beginners-pid-introduction/

if __name__ == "__main__":
    try:
        logging.basicConfig(format="%(asctime)s %(name)s \
            %(levelname)s %(message)s", level = logging.CRITICAL)
        mp.log_to_stderr(logging.DEBUG)
        TemperatureQueue = mp.Queue()
        StatusQueue = mp.Queue()
        MovementQueue = mp.Queue()
        myRiceCooker = RiceCookerController(phase_cycle_in_sec=PHASE_LENGTH, 
                        split=True, StatusQueue=StatusQueue, 
                        MovementQueue=MovementQueue)
        myRiceCooker.start()
        myTemperatureProvider = TemperatureProvider(phase_cycle_in_sec=PHASE_LENGTH,
                        Resolution=12, TemperatureQueue=TemperatureQueue)
        myTemperatureProvider.start()
        myAgent = Agent(kP=1.075, kI=0.01, kD=80, target_temp=54, 
                        target_duration=300, TemperatureQueue=TemperatureQueue,
                        StatusQueue=StatusQueue, MovementQueue=MovementQueue, 
                        label="Fillet_Mignon_Amazon_b_")
        myAgent.start()
        myRiceCooker.join()
        myTemperatureProvider.join()
        myAgent.join()
    except:
        logging.error("Exception occured", exc_info=True)

# TODO
# develop the function self.shutdown() in each of the classes
# Check if there's any scikit learn or scipy for PID ?
# Develop the shutdown function in the Agent
# use the property decorator to make the code better
# Learn how to use the Logger
# Experiment reducing the cycle length to 1 second - better results?
# Make it multi-threading
# Add a rounding factor of 4 to outcome_P, outcome_I, outcome_D
# Take more samples
# Read Arduino PID library documentation
# find a way to query the temperature sensor faster
# check if there's a way to query the tuya stuff faster
# Add a sound buzzer
# make it buzz if there's an error
# make it buzz when the timer is over
# make it buzz when it's ready to add the egg
# Auto start the script from the device without an IDE
# Add buttons that can change the duration and target temp
# Upload logs automatically to Azure blob
# Keep temperature at same level for the experiment duration
# Find out how to force Tuya switch off in case of exception 
# Rename the file of the experiment/log
# Add UUID to experiment CSV
# Force shutdown of Tuya if keyboard exit
# figure out if i can increase the resolution of the temperature sensor
# shorten timestamp in logger
# make the time correct (on the timezone)
# in Analyzer, change X axis to time (minutes)
# Analyzer - draw a line that shows the peaks (except first peak)
# Analyzer - histogram of temperatures
# change the logic of turning switches on and off so that it is based on a SMA
# add to git

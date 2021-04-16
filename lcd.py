
import atexit
from agentenvironment_pid import Environment, Agent


# https://www.seriouseats.com/2013/10/sous-vide-101-all-about-eggs.html
# https://www.wescottdesign.com/articles/pid/pidWithoutAPhd.pdf
# http://brettbeauregard.com/blog/2011/04/improving-the-beginners-pid-introduction/

def run():
    global myEnv
    myEnv = Environment(phase_cycle_in_sec=1)
    myAgent = Agent(kP=1, kI=0.01, kD=100, target_temp=60, 
                    target_duration=60, Environment=myEnv, label="test")
    while True:
        myAgent.take_step()


def exit_handler():
    myEnv.shutdown()
    print("end")


if __name__ == "__main__":
    atexit.register(exit_handler)
    try:
        run()        
    except:
        exit_handler()

        

# TODO
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

# download package apm.py from APMonitor.com
from apm import *
# define server and application name
s = 'http://byu.apmonitor.com'

## standard model in APMonitor
a = 'tune1'
# clear any prior application with that name
apm(s,a,'clear all')
# load model and data file
apm_load(s,a,'model1.apm')
csv_load(s,a,'data.csv')
# dynamic optimization
apm_option(s,a,'nlc.imode',6)
apm_option(s,a,'nlc.nodes',3)
apm_option(s,a,'nlc.cv_type',1)
apm_option(s,a,'nlc.mv_type',0)
# manipulated variable (u)
apm_info(s,a,'MV','u')
apm_option(s,a,'u.dcost',0.1)
apm_option(s,a,'u.dmax',2)
apm_option(s,a,'u.status',1)
apm_option(s,a,'u.fstatus',0)
apm_option(s,a,'u.upper',8)
apm_option(s,a,'u.lower',0)
# state variable
apm_info(s,a,'SV','x')
# controlled variable (x)
apm_info(s,a,'CV','y')
apm_option(s,a,'y.status',1)
apm_option(s,a,'y.fstatus',0)
apm_option(s,a,'y.sphi',7.1)
apm_option(s,a,'y.sp',7.0)
apm_option(s,a,'y.splo',6.9)
apm_option(s,a,'y.tau',2)
apm_option(s,a,'y.tr_init',2)
apm_option(s,a,'y.tr_open',1)
apm_option(s,a,'y.wsphi',200)
apm_option(s,a,'y.wsplo',200)

# solve and retrieve solution
output = apm(s,a,'solve')
print(output)
z1 = apm_sol(s,a)



## state space continuous model in APMonitor
a = 'tune2'
# clear any prior application with that name
apm(s,a,'clear all')
# load model and data file
apm_load(s,a,'model2.apm')
csv_load(s,a,'data.csv')
# dynamic optimization
apm_option(s,a,'nlc.imode',6)
apm_option(s,a,'nlc.nodes',3)
apm_option(s,a,'nlc.diaglevel',0)
apm_option(s,a,'nlc.mv_type',0)
apm_option(s,a,'nlc.cv_type',1)
# manipulated variable (u)
apm_info(s,a,'MV','u')
apm_option(s,a,'u.dcost',0.1)
apm_option(s,a,'u.dmax',2)
apm_option(s,a,'u.status',1)
apm_option(s,a,'u.fstatus',0)
apm_option(s,a,'u.upper',8)
apm_option(s,a,'u.lower',0)
# state variable
apm_info(s,a,'SV','x')
# controlled variable (x)
apm_info(s,a,'CV','y')
apm_option(s,a,'y.status',1)
apm_option(s,a,'y.fstatus',0)
apm_option(s,a,'y.sphi',7.1)
apm_option(s,a,'y.sp',7.0)
apm_option(s,a,'y.splo',6.9)
apm_option(s,a,'y.tau',2)
apm_option(s,a,'y.tr_init',2)
apm_option(s,a,'y.tr_open',1)
apm_option(s,a,'y.wsphi',200)
apm_option(s,a,'y.wsplo',200)

# solve and retrieve solution
output = apm(s,a,'solve')
print(output)
z2 = apm_sol(s,a)



## state space continuous model in APMonitor
a = 'tune3'
# clear any prior application with that name
apm(s,a,'clear all')
# load model and data file
apm_load(s,a,'model3.apm')
csv_load(s,a,'data.csv')
# dynamic optimization
apm_option(s,a,'nlc.imode',6)
apm_option(s,a,'nlc.nodes',2)
apm_option(s,a,'nlc.diaglevel',0)
apm_option(s,a,'nlc.mv_type',0)
apm_option(s,a,'nlc.cv_type',1)
# manipulated variable (u)
apm_info(s,a,'MV','u')
apm_option(s,a,'u.dcost',0.1)
apm_option(s,a,'u.dmax',2)
apm_option(s,a,'u.status',1)
apm_option(s,a,'u.fstatus',0)
apm_option(s,a,'u.upper',8)
apm_option(s,a,'u.lower',0)
# state variable
apm_info(s,a,'SV','x')
# controlled variable (x)
apm_info(s,a,'CV','y')
apm_option(s,a,'y.status',1)
apm_option(s,a,'y.fstatus',0)
apm_option(s,a,'y.sphi',7.1)
apm_option(s,a,'y.sp',7.0)
apm_option(s,a,'y.splo',6.9)
apm_option(s,a,'y.tau',2)
apm_option(s,a,'y.tr_init',2)
apm_option(s,a,'y.tr_open',1)
apm_option(s,a,'y.wsphi',200)
apm_option(s,a,'y.wsplo',200)

# solve and retrieve solution
output = apm(s,a,'solve')
print(output)
z3 = apm_sol(s,a)


# plot solution
import matplotlib.pyplot as plt
plt.figure(1)
plt.subplot(211)
plt.plot(z1['time'],z1['u'],'r-',linewidth=2)
plt.plot(z2['time'],z2['u'],'b--',linewidth=2)
plt.plot(z3['time'],z3['u'],'k-x',linewidth=2)
plt.legend('u')
plt.ylabel('MV')
plt.xlabel('Time (sec)')

plt.subplot(212)
plt.plot(z1['time'],z1['y'],'r-',linewidth=2)
plt.plot(z2['time'],z2['y'],'b--',linewidth=2)
plt.plot(z3['time'],z3['y'],'k-x',linewidth=2)
plt.plot(z1['time'],z1['y.tr_hi'],'r--',linewidth=1)
plt.plot(z1['time'],z1['y.tr_lo'],'r--',linewidth=1)
plt.plot(z1['time'],z3['y.tr_hi'],'g--',linewidth=1)
plt.plot(z1['time'],z3['y.tr_lo'],'g--',linewidth=1)
plt.legend(['ODE Model','Continuous SS','Discrete SS','Set Point'])
plt.ylabel('CV')
plt.xlabel('Time (sec)')

plt.show()

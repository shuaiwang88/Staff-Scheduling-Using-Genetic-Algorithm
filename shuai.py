#-*- coding: utf-8 -*-
"""
Created on Tue Jan 30
Genetic algorithm
@author: shuai wang
"""
import numpy as np
# import matplotlib.pyplot as plt
from time import time
import random as rd
t = time()
#----------------------------


# INPUT
# minRest = 0.5 * 2
# maxRest = 2.5 * 2

rest_time=1
LastShiftEndTime = 24*7 +8 # 176 next monday's morning 8am
day_rest=12
days = np.arange(7)
time_8h=8
off_days_8h=2

# hour_size=24*7

#shift
numWorkers = 4 #10
cost_fulltime= 20 #
#demand of workers
demand = np.loadtxt("Demand_Test.csv", delimiter='\t', usecols=range(1,25),dtype=np.int)
hourly_staff_handle=20

under_cover_cost=40


# Genetic algorithm parameters
popuSize = 3 # 200   5   10
probCross = 0.8 #0.8
mutaSize = 2
probMutation = 0.1
numElitism = 20 #20    2    4
probFullTime = 0.5
maxIt = 300 #300

#------ generate shift for each worker on weekly basis (0,7*24)
def shift8h():
    shift = np.zeros((7, off_days_8h), dtype=int)
    twodaysoff = np.sort(np.random.choice(range(7), off_days_8h, replace=False)) # random 2 dif number
    # twodaysoff=[0,3] # twodaysoff=[1,6] # wodaysoff=[0,1] # twodaysoff=[1,2]
    print("twodaysoff:", twodaysoff)
    work_days = np.setdiff1d(days, twodaysoff) # 5 work days

    for i in work_days:
        for j in twodaysoff:
            if j - i != 1: # if the day is NOT the previous day before the off-days
                starttime1 = np.random.randint(0, 23) + i * 24
                endtime = starttime1 + time_8h
                shift[i] = (starttime1, endtime)
                # else:
            if j - i == 1 and j != 6: # if the days is the day before the off-days
                starttime1 = np.random.randint(0, 16) + i * 24 #not start work withing 8 hours before midnight of off
                endtime = starttime1 + time_8h
                shift[i] = (starttime1, endtime)
                # add
                if starttime1 - shift[i - 1][1] < 12:  # 12 hours break btw work days
                    print('yes')
                    # shift[i-1][0]=np.random.randint(0,starttime1-12) +i*24
                    shift[i - 1][0] = np.random.randint(0, 10) + i * 24
                    # print(shift[i-1][0])
                    shift[i - 1][1] = shift[i - 1][0] + time_8h

                    # print(shift[i])
            if j == 6:
                shift[j] = [0, 0]
            if j == 1:
                shift[j - 1][0] = (np.random.randint(0, 17))  # 0..16 24-8
                shift[j - 1][1] = shift[j - 1][0] + time_8h
    return shift

def modify_shift8h():
    mod_shift8h = shift8h()
    #print("old:", a)
    for i in range(1, len(mod_shift8h) - 1):
        if abs(mod_shift8h[i][0] - mod_shift8h[i - 1][1]) < 12 and np.array_equal(mod_shift8h[i],
                                                    np.array([0, 0])) == False:  # and mod_shift8h[i][0]-a[i-1][1]!=0:
            mod_shift8h[i - 1][1] = mod_shift8h[i][0] - 12 - np.random.randint(0, 4)
            mod_shift8h[i - 1][0] = mod_shift8h[i - 1][1] - time_8h
    return mod_shift8h
    # if a[i][0]-a[i-1][1] <0:
    #     a[i-1]=[0,0]
    # print("new:",a)
#----------
a=modify_shift8h()
b=modify_shift8h()
c=modify_shift8h()
d=modify_shift8h()

e=modify_shift8h()
f=modify_shift8h()
g=modify_shift8h()
h=modify_shift8h()
gen1=(a,b,c,d)
gen2=(e,f,g,h)
# array([[  0,   0],
#        [ 45,  53],
#        [ 62,  70],
#        [ 84,  92],
#        [  0,   0],
#        [122, 130],
#        [153, 161]])

#-------  mapping shift like 2:10 26:34 until 24*7+8 to 0,1 #
def integer2binaryShift(workerInteger):
    workerBinary= np.zeros(LastShiftEndTime,dtype=np.int)
    # hour_size_arr=workerInteger
    # print(hour_size_arr)
    for hr in workerInteger:
        # print(hr)
        if np.array_equal(hr, [0, 0]) == False:
            workerBinary[hr[0]:hr[0] + time_8h + 1] = 1  # +8h (index +1)
    # print(workerBinary)
    return workerBinary
#------
integer2binaryShift(b)

# b=popuInteger[0][j,:]


numWorkers = 3 #10
#Generation of random population, complying with worker restrictions
popuInteger = np.zeros((popuSize,numWorkers,7,2),dtype=np.int)
popuBinary = np.zeros((popuSize,numWorkers,LastShiftEndTime),dtype=np.int)

for i in range(0,popuSize):
    for j in range(0,numWorkers):
            popuInteger[i, j, :] = modify_shift8h()





# this return  # popusize of shift2demand
# def shift2demand(genInteger):
#     # So this makes a matrix #workers and # 38 which is length of the day
#     genBinary = np.zeros([popuSize,numWorkers,LastShiftEndTime],dtype=np.int)
#     genBinary_sum=np.zeros([popuSize,LastShiftEndTime],dtype=np.int)
#     for i in range(popuSize):
#         for j in range(numWorkers): #numwWorkers-1
#             # genBinary[i][j,:] = integer2binaryShift(popuInteger[i][j,:])
#             genBinary[i][j, :] = integer2binaryShift(genInteger[i][j, :])
#         genBinary_sum[i]=sum(genBinary[i], 0) #* hourly_staff_handle#20
#
#     return genBinary_sum




 # return for each worker in the popusize
def shift2demand(genInteger):
    genBinary = np.zeros([numWorkers,LastShiftEndTime],dtype=np.int)
    genBinary_sum = np.zeros(LastShiftEndTime, dtype=np.int)
    for i in range(numWorkers): #numwWorkers-1
        genBinary[i] = integer2binaryShift(genInteger[i])
    genBinary_sum=sum(genBinary, 0) #* hourly_staff_handle#20
    return genBinary_sum


shift2demand(popuInteger[0])

######



## --------------  objective
# convert 168 demand to 168+8 demand
demand_flat = demand.flatten()
demand_overweek = np.append(demand_flat,demand_flat[0:8])
demand_require_workers=np.ceil(demand_overweek/hourly_staff_handle)

####


max_under_coverage= np.sum(np.maximum(demand_require_workers,
                                      demand_require_workers-numWorkers))

under_cover= demand_require_workers - numWorkers

xxx=popuInteger[0][0] #shuai
xxx
shift2demand(xxx)

def computeFitness(genInteger):
    # shift_hourly_capability=shift2demand(genInteger)* hourly_staff_handle
    cost_worker_week = np.sum(cost_fulltime * shift2demand(genInteger))
    sum_under_cover = np.sum(under_cover[under_cover>0])
    under_cover_penalty = under_cover_cost * sum_under_cover
    total_cost = cost_worker_week + under_cover_penalty
    return total_cost

computeFitness(genInteger)

genError = float(np.sum(abs(getGenWorkersHalfHours(genInteger) -
                            workersHalfHours))) / float(maxError)


#----  set up a middle point and crossover/swap, more methods see literature

def crossover(gen1,gen2): # one gen is the all shifts for all the workers
    pointOfCross = np.random.randint(1,numWorkers-2)
    return(np.concatenate((gen1[0:pointOfCross,:],gen2[pointOfCross:numWorkers,:])),
           np.concatenate((gen2[0:pointOfCross,:],gen1[pointOfCross:numWorkers,:])))

crossover(gen1,gen2)





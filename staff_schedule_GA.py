#-*- coding: utf-8 -*-
"""
Created on Tue Jan 30
Genetic algorithm
@author: shuai wang
"""

###################
import numpy as np
import matplotlib.pyplot as plt
from time import time
import random as rd
t = time()
#----------------------------


# INPUT
# minRest = 0.5 * 2
# maxRest = 2.5 * 2

# rest_time=1
LastShiftEndTime = 24*7# +8 # 176 next monday's morning 8am
hours_168 =168
day_rest=12
days = np.arange(7)
time_8h=8


off_days_8h=2
off_days_8h_parttime=5



# hour_size=24*7

#shift
numWorkers = 27 #25
num_fulltime = 25# 23
num_parttime= 2 #2

cost_fulltime= 20 #
shift_start_end = 2  # have a start and a end integer each day

full_time_total_hour = 42
max_working_days = 5

work_hours_options = [6, 8, 10]
weights = [0.2, 0.7, 0.1]

#demand of workers
# demand = np.loadtxt("Demand_Test.csv", delimiter='\t', usecols=range(1,25),dtype=np.int)
# demand = np.genfromtxt("2015-11-26.csv", delimiter=',', skip_header =1,usecols=3,dtype=np.int)
demand = np.genfromtxt("2016-06-20.csv", delimiter=',', skip_header =1,usecols=3,dtype=np.int)

demand=np.array(np.split(demand,7))
demand_daily = np.sum(demand,axis=1)
demand_order= np.argsort(demand_daily, axis=0)[::-1]+1 # daily demand in descending order


hourly_staff_handle=1

under_cover_cost=400

over_cover = -50
under_cover = 50
# Genetic algorithm parameters
popuSize =100# 200   5   10
probCross = 0.8 #0.8
mutaSize = 1 #2
probMutation = 0.2 # 0.2
numElitism = 20 #20    2    4
maxIt =200 #300

#------ generate shift for each worker on weekly basis (0,7*24)

def shift_generator():
# for i in range(10):
#     work=np.array([1,2,3,4,5,6,7],np.int)
#     work_random=np.random.permutation(work)
    work_random = demand_order # assign high demand with priority
    if  np.random.random_sample()> 0.7:
        work_random=np.random.permutation(work_random)

    work_random_binary=np.zeros(7,dtype=np.int)
    work_hour_daily = np.zeros(7, dtype=np.int)
    work_total_hours=0

    # print(work_random)
    for i in range(len(work_random)):
        hour_rand = np.random.choice(work_hours_options, p=weights)
        work_hour_daily[work_random[i] - 1] = hour_rand

        work_random_binary[work_random[i] - 1] = 1  # new
        work_total_hours += hour_rand

        if work_total_hours == 34:
            hour_rand = np.random.choice(work_hours_options, p=[0.3, 0.7, 0])
            work_hour_daily[work_random[i+1] - 1] = hour_rand
            work_random_binary[work_random[i+1] - 1] = 1  # new
            break

        if (work_total_hours == 36) and (np.sum(work_random_binary) < max_working_days):  # no 6 days
            hour_rand = np.random.choice(work_hours_options, p=[1, 0, 0])
            work_hour_daily[work_random[i+1] - 1] = hour_rand
            work_random_binary[work_random[i+1] - 1] = 1  # new
            break

        if work_total_hours > full_time_total_hour:
            work_hour_daily[work_random[i] - 1] = 0
            work_total_hours -= hour_rand
            work_random_binary[work_random[i] - 1] = 0
            break
    # print(work_random_binary)
    # print(work_hour_daily)

# Pick start time for the first day
#     start = np.zeros(7, np.int16) # 7
#     end = np.zeros(7, np.int16)  # 7
    start = np.zeros(8, np.int16)  # 7
    end = np.zeros(8, np.int16)  # 7

    work_random_binary = np.array((work_random_binary), dtype=bool)

    first_day = np.argmax(work_random_binary)
    first_day = np.argmax(work_random_binary)
    start[first_day] = np.random.randint(first_day * 24, (first_day + 1) * 24)# old
    # start[first_day] = np.random.randint(first_day * 24, (first_day + 1) * 24-15)
    end[first_day] = start[first_day] + work_hour_daily[first_day]
    # Pick start times for the following days
    # work_hour_daily

    for i in range(len(work_hour_daily)):
        if work_random_binary[i] == True:
            while (start[i] - start[i - 1]) < 12 + work_hour_daily[i - 1]:
                start[i] = np.random.randint(i * 24, (i + 1) * 24 - 1)
                end[i] = start[i] + work_hour_daily[i]
# this is cyclic schedule
                if end[i] >168:
                    start[i+1] = 0
                    end[i+1] = end[i] - 168
                    end[i]=168


    return np.array(list(zip(start, end)))

# for i in range(100):
#     print(shift_generator())
#
# def part_time_8h(): # part_time_8h(time_8h)
#     shift = np.zeros((7, shift_start_end), dtype=np.int)
#     random1 = np.random.randint(hours_168 - 12 - 8)
#     # print(random1)
#     if random1>(23+time_8h): # 31, random1 can be like 11, so random2 can be only > random1
#         if np.random.random_sample()>0.5: ## day1 either earlier or later
#             random2 = np.random.randint(random1 + 8 + 12, hours_168)
#         else:
#             random2 = np.random.randint(0,random1-8-12)
#     else:
#         random2 = np.random.randint(random1 + 8 + 12, hours_168)
#
#     random1_day = random1 // 24
#     random2_day = random2 // 24
#
#     shift[random1_day], shift[random2_day] = [random1,random1+time_8h] , [random2,random2+time_8h]
#
#     return shift


# code
def part_time_8h(): # part_time_8h(time_8h)

    shift = np.zeros((8, shift_start_end), dtype=np.int) #7
    hour_rand1 = np.random.choice(work_hours_options, p=weights)
    random1 = np.random.randint(hours_168 - 12 - hour_rand1-10)
    # print(random1)
    if random1>(23+hour_rand1): # 31, random1 can be like 11, so random2 can be only > random1
        if np.random.random_sample()>0.5: ## day1 either earlier or later
            random2 = np.random.randint(random1 + hour_rand1 + 12, hours_168-10)
        else:
            random2 = np.random.randint(0,random1-hour_rand1-12)
    else:
        random2 = np.random.randint(random1 + hour_rand1 + 12, hours_168)
    hour_rand2 = np.random.choice(work_hours_options, p=weights)

    random1_day = random1 // 24
    random2_day = random2 // 24

    shift[random1_day], shift[random2_day] = [random1,random1+hour_rand1] , [random2,random2+hour_rand2]

    return shift

#
#
#
#
#
# for i in range(100):
#     print(part_time_8h())

xxx=np.array(
[[ 1  ,9],
 [  0  , 0],
 [  0  , 0],
 [ 85 , 91],
 [106, 116],
 [132, 140],
 [166, 168],
 [0,6]])


#-------  mapping shift like 2:10 26:34 until 24*7+8 to 0,1 #
def integer2binaryShift(workerInteger):
    workerBinary= np.zeros(LastShiftEndTime,dtype=np.int)
    # hour_size_arr=workerInteger
    # print(hour_size_arr)
    for hr in workerInteger:
        # print(hr)
        if np.array_equal(hr, [0, 0]) == False:
            # workerBinary[hr[0]:hr[0] + time_8h + 1] = 1  # +8h (index +1)
            # workerBinary[hr[0]:hr[0] + time_8h] = 1  # +8h (index +1)
            # workerBinary[hr[0]:hr[1]] = 1 # old for non cyclic
            workerBinary[hr[0]:hr[1]] += 1
        # print(
    # print(workerBinary)
    return workerBinary

integer2binaryShift(xxx)

#------
b=part_time_8h()
b=shift_generator()

integer2binaryShift(b)

# b=popuInteger[0][j,:]


#Generation of random population, complying with worker restrictions
# popuInteger = np.zeros((popuSize,numWorkers,7,2),dtype=np.int)       # before cyclic
popuInteger = np.zeros((popuSize,numWorkers,8,2),dtype=np.int)
popuBinary = np.zeros((popuSize,numWorkers,LastShiftEndTime),dtype=np.int)

auxPopuInteger = popuInteger
# for i in range(0,popuSize):
#     for j in range(0,numWorkers):
#             popuInteger[i, j, :] = modify_shift8h()

for i in range(0, popuSize):
    for j in range(0, num_fulltime):
        popuInteger[i, j, :] = shift_generator()

    for k in range(0, num_parttime):
        popuInteger[i, num_fulltime+k, :] = part_time_8h() # add part time




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
    # genBinary_sum = np.zeros(LastShiftEndTime, dtype=np.int)
    for i in range(numWorkers): #numwWorkers-1
        genBinary[i] = integer2binaryShift(genInteger[i])
        # print(genBinary[i])
    genBinary_sum=sum(genBinary, 0) #* hourly_staff_handle#20
    return genBinary_sum

xx=shift2demand(popuInteger[0])

######



## --------------  objective
# convert 168 demand to 168+8 demand
demand_flat = demand.flatten()
# demand_overweek = np.append(demand_flat,demand_flat[0:8]) ## add up mondays's fist 8 hours
# demand_require_workers=np.ceil(demand_overweek/hourly_staff_handle)
demand_require_workers=demand_flat
# max_under_coverage= np.sum(np.maximum(demand_require_workers,
#                                       demand_require_workers-numWorkers))

# num_undercover= demand_require_workers - numWorkers # this is wrong



## THIS IS FULL TIME COST
def computeFitness(genInteger):
    # shift_hourly_capability=shift2demand(genInteger)* hourly_staff_handle
    num_undercover = demand_require_workers - shift2demand(genInteger)
    cost_worker_week = np.sum(cost_fulltime * shift2demand(genInteger))

    sum_under_cover = np.sum(num_undercover[num_undercover>0])
    under_cover_penalty = under_cover_cost * sum_under_cover

    total_cost = cost_worker_week + under_cover_penalty


    return np.int(total_cost)

computeFitness(popuInteger[0,:,:])

#
# aa=popuInteger[0,:,:]
# num_undercover = demand_require_workers - shift2demand(aa)
# cost_worker_week = np.sum(cost_fulltime * shift2demand(aa))
# sum_under_cover = np.sum(num_undercover[num_undercover > 0])



#----  set up a middle point and crossover/swap, more methods see literature

def crossover(gen1,gen2): # one gen is the all shifts for all the workers
    pointOfCross = np.random.randint(1,numWorkers-2)
    # print(pointOfCross)
    return(np.concatenate((gen1[0:pointOfCross,:],gen2[pointOfCross:numWorkers,:])),
           np.concatenate((gen2[0:pointOfCross,:],gen1[pointOfCross:numWorkers,:])))

crossover(popuInteger[0],popuInteger[1])


#-----   mutation combine full time and part time
def mutation(gen):
    mutationIdx_full = np.random.permutation(num_fulltime) # last
    mutationIdx_full = mutationIdx_full[0:mutaSize]
    # print(mutationIdx_full)
    for i in mutationIdx_full:
        gen[i,:] = shift_generator()

    for j in range(num_fulltime,num_fulltime+num_parttime): # add parttime
        if np.random.random_sample()>0.5:
            gen[j,:] = part_time_8h()
    return gen
#

#- fitness of a population


# Fitness function of each gen
popuFitness = np.zeros([popuSize],dtype=np.int)
for i in range(popuSize):
    popuFitness[i] = computeFitness(popuInteger[i, :, :])

cumulativeFitness = np.cumsum(popuFitness)


it=0
minPopuFitness = np.zeros([maxIt],)
auxPopuInteger = popuInteger


# save results:
# results_iter = np.zeros(maxIt,dtype=np.int)

while it < maxIt:
    sortedIndexPopuFitness = np.argsort(popuFitness)  # low --> high
    # best_index_numElitism = popuSize - numElitism + 1 # shuai this is best # max value
    # auxPopuInteger[0:numElitism - 1, :, :] = popuInteger[sortedIndexPopuFitness[best_index_numElitism:], :, :]
    best_index_numElitism = numElitism-1 #+1
    auxPopuInteger[0:numElitism - 1, :, :] = popuInteger[sortedIndexPopuFitness[0:best_index_numElitism], :, :]


# ##old crossover
    numCrossPairs = np.random.binomial((popuSize-numElitism)/2,probCross)  # print 'numCrossPairs',numCrossPairs #shuai
    numNoCrossGenes = popuSize - 2*numCrossPairs - numElitism
    #
    for k in range(0, numCrossPairs - 1):
        selected1 = np.argmax(cumulativeFitness >= np.random.random() * cumulativeFitness[-1])
        selected2 = np.argmax(cumulativeFitness >= np.random.random() * cumulativeFitness[-1])

        cross = crossover(popuInteger[selected1, :, :], popuInteger[selected2, :, :])
        auxPopuInteger[numElitism + 2 * k, :, :] = cross[0]
        auxPopuInteger[numElitism + 2 * k + 1, :, :] = cross[1]
        # this updates the auxPopuInteger every two do an append

    for k in range(0, numNoCrossGenes - 1):
        selected = np.argmax(cumulativeFitness >= np.random.random() * cumulativeFitness[-1])
        auxPopuInteger[numElitism + 2 * numCrossPairs + k, :, :] = popuInteger[selected, :, :]
            # again append more solution to the pool.

#REDO CROSSOVER:

    # numCrossPairs = np.random.binomial((popuSize - numElitism) / 2, probCross) # from old
    # worst_PopuFitness = np.max(popuFitness)
    # worst_PopuFitness_ind = np.argmax(popuFitness)
    #
    # for k in range(0, numCrossPairs - 1): # from old
    #     parents=np.random.choice(numWorkers,2,replace=False) # 1 pairs 2; 2 pairs 4
    #     kids1=parents[0:2]
    #     # kids2=parents[2:4]
    #     cross1 = crossover(popuInteger[kids1[0], :, :], popuInteger[kids1[1], :, :])
    #     kids1_fit1=computeFitness(cross1[0])
    #     kids1_fit2=computeFitness(cross1[1])
    #     kids1_best_ind = np.argmin((kids1_fit1,kids1_fit2))
    #     kids1_best = np.max((kids1_fit1,kids1_fit2))
    #
    #     if kids1_best > worst_PopuFitness:
    #         # auxPopuInteger[numElitism+1,:,:] = cross1[kids1_best_ind]
    #         auxPopuInteger[numElitism + 2*k, :, :] = cross1[kids1_best_ind] # from old
    #




## Mutation old
    # numMutation = np.random.binomial(popuSize,probMutation)
    # # print numMutation #shuai
    # indexToMutate = np.random.randint(numElitism,popuSize-1,numMutation)
    # for k in range (0,numMutation-1):
    #        auxPopuInteger[indexToMutate[k],:,:] = mutation(auxPopuInteger[indexToMutate[k],:,:]);



# ## REDO MUTATION:
    numMutation = np.random.binomial(popuSize, probMutation)
    indexToMutate = np.random.randint(numElitism, popuSize - 1, numMutation)
    for k in range(0, numMutation - 1):
        auxPopuInteger[indexToMutate[k], :, :] = mutation(auxPopuInteger[indexToMutate[k], :, :]);
# below is to reduce workers
        undercover_num = demand_require_workers - shift2demand(auxPopuInteger[indexToMutate[k]])
        sum_overcover= np.sum(undercover_num[undercover_num<0])
        sum_undercover = np. sum(undercover_num[undercover_num>0]) # shuai



        # if  sum_overcover < over_cover :
        if sum_overcover < over_cover or sum_undercover < under_cover:
            # print('sum_overcover',sum_overcover)
            # print('sum_undercover;', sum_undercover)
            ran_worker=np.random.randint(numWorkers)
            auxPopuInteger[indexToMutate[k],ran_worker, :] = np.zeros((8, shift_start_end), dtype=np.int) #7



    # final assiangment
    popuInteger = auxPopuInteger



    # Fitness function
    for i in range(popuSize):
        popuFitness[i] = computeFitness(popuInteger[i, :, :])

    cumulativeFitness = np.cumsum(popuFitness)
    bestSolInd = np.argmin(popuFitness)
    minPopuFitness[it] = popuFitness[bestSolInd]
    # print(minPopuFitness)
    print (minPopuFitness[it]) # shuai
    # print it #shuai 2
    print(it)
    # results_iter[it] = minPopuFitness[it] #s
    it = it+1



bestSolution = popuInteger[bestSolInd,:,:]
solution = shift2demand(popuInteger[bestSolInd,:,:])
print (minPopuFitness[it-1])

t1 = time()
print ('time',t1-t)


print ('bestSolution:',  bestSolution)
num_ass = 0

for i in bestSolution:
    if np.array_equal(i,np.zeros((7, shift_start_end), dtype=np.int)) == False:
        num_ass += 1

print('assigned # workers:', num_ass)

print('demand:',demand_require_workers )
print('total_demand',np.sum(demand_require_workers))
print('bestsoltuon_cover:', shift2demand(bestSolution))

sol_uncover =demand_require_workers - solution
print ("undercover,",sol_uncover)

print("undercover_total",np.sum(sol_uncover[sol_uncover>0]) )
print("overcover_total",np.sum(sol_uncover[sol_uncover<0]) )

#
# min_obj=int(minPopuFitness[-1])
# plt.plot(range(maxIt),results_iter)
#
# plt.title("objective: %d" % min_obj)
# plt.show()
# plt.savefig('result_20151126.png')
# plt.close()
#
#

#
#
# firstShift = bestSolution[:]
# # secondShift = bestSolution[:,2:4]
# #broken_barh?
# y = np.arange(numWorkers)
# x = np.arange(LastShiftEndTime)
# ax1 = plt.subplot(211)
# plt.barh(y,firstShift[:,1]-firstShift[:,0],0.5,firstShift[:,0],hold=True)
# # plt.barh(y,secondShift[:,1]-secondShift[:,0],0.5,secondShift[:,0],hold=True)
# plt.xticks(x)
# plt.grid()
# plt.subplot(212, sharex=ax1)
# plt.grid()
# plt.bar(x*0.5+8,solution-demand_require_workers,width=0.5)
# plt.show()
#
#
#

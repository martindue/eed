#approac med linspace

import numpy as np
import matplotlib.pyplot as plt


import numpy as np
from scipy.special import expit

def sigmoid_linspace(start, end, num):
    """
    Generate `num` values between `start` and `end`, spaced according to a sigmoid function.
    
    Parameters:
    - start: The starting value of the sequence.
    - end: The end value of the sequence.
    - num: Number of values to generate.
    - gain: The gain (slope) of the sigmoid function. Higher values make the sigmoid steeper.
    
    Returns:
    - A numpy array of `num` values spaced according to a sigmoid function.
    """
    # Generate `num` values linearly spaced between -gain and gain
    gain = 100
    linear_space = np.linspace(-gain, gain, num)
    
    # Apply the sigmoid function to these values
    sigmoid_space = expit(linear_space)
    
    # Scale and shift the sigmoid values to be between `start` and `end`
    scaled_sigmoid_space = start + (end - start) * sigmoid_space
    scaled_sigmoid_space[0] = start
    scaled_sigmoid_space[-1] = end


    return scaled_sigmoid_space


num_saccs = 30
weib_shape = 1.78 # found by fitting a weibull distribution to validation fixation durations
weib_scale = 6.48 # found by fitting a weibull distribution to validation fixation durations
freq = 1
arr = np.array([0])
for i in range(num_saccs):
    fix_length = np.random.weibull(2.57)*151
    fix = np.full(int(fix_length*freq), arr[-1])
    arr = np.append(arr, fix)

    sacc_length = np.random.weibull(weib_shape)*weib_scale
    if np.random.rand() < 0.5:
        sacc_length = -sacc_length
    
    sacc = np.linspace(arr[-1], arr[-1] + sacc_length, np.abs(int(sacc_length*freq)))
    arr = np.append(arr, sacc)

plt.plot(arr,"r+-")
plt.show()















import numpy as np
import matplotlib.pyplot as plt
num_saccs = 5

x_arr = np.array([0])
y_arr = np.array([0])
for i in range(num_saccs):
    fix_length = np.random.randint(150, 1000)
    sacc_length = np.random.randint(20, 150)
    last_x = x_arr[-1]
    fix_samples = np.r_[last_x:last_x + fix_length]

    x_arr = np.append(x_arr, fix_samples)


    last_y = y_arr[-1]
    y_arr = np.append(y_arr, np.full(fix_length, last_y))

    sacc_samples_x = np.r_[x_arr[-1]:x_arr[-1]+sacc_length]


    if np.random.rand() < 0.5:
        sacc_samples_y = np.r_[y_arr[-1]:y_arr[-1] + sacc_length]
    else:
        sacc_samples_y = np.r_[y_arr[-1]:y_arr[-1]-sacc_length]
    x_arr = np.append(x_arr, sacc_samples_x)
    y_arr = np.append(y_arr, sacc_samples_y)

plt.plot(x_arr, y_arr)
plt.show()





    































import numpy as np

n = 5
weib_shape = 5
x_list = [0]
y_list = [0]

for i in range(n):
    fixation_length = np.random.weibull(weib_shape)
    x_fix = x_list[2*i] + fixation_length
    y_fix = y_list[2*i]
    print("x_fix for iteration :",i,"is",x_fix)
    fix_samples = np.linspace(x_list[2*i], x_fix, 10)
    x_list.append(x_fix)
    y_list.append(y_fix)

    saccade_length = np.random.weibull(weib_shape, 1).item()

    x_sac = x_fix + saccade_length*0.1 
    print("x_sac for iteration :",i,"is",x_sac)
    saccade_length = np.random.choice([-1, 1], size=1).item() * saccade_length

    y_sac = y_fix + saccade_length
    
    x_list.append(x_sac)
    y_list.append(y_sac)

    # Add normal noise to x_list and y_list
#x_list = np.array(x_list) + np.random.normal(0, 0.01, len(x_list))
#y_list = np.array(y_list) + np.random.normal(0, 0.01, len(y_list))

import matplotlib.pyplot as plt
print(x_list)
print(y_list)
plt.plot(x_list, y_list)
plt.show()
    



import numpy as np
n = 5

weib_shape = 5

x_list = [0,0]
y_list = [0,0]
for i in range(1,n):
    #fixation_length = np.random.weibull(weib_param, 1)
    y_new = np.random.weibull(weib_shape, 1).item()
    y_new = np.random.choice([-1, 1], size=1).item() * y_new
    y_new = y_list[i-1] + y_new
    y_list.append(y_new)

    saccade_length = np.random.weibull(weib_shape, 1).item()*0.1
    x_new = x_list[i-1] + saccade_length

    #x_new = x_list[i-1] + np.random.weibull(weib_param, 1).item()
    x_list.append(x_new)

    fixation_length = np.random.weibull(weib_shape, 1).item()
    x_list.append(x_list[i] + fixation_length)
    y_list.append(y_new)

import matplotlib.pyplot as plt
print(x_list)
print(y_list)
plt.plot(x_list, y_list)
plt.show()

print(np.random.weibull(weib_shape, 1000).item())

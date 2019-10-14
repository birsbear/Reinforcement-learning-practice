import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

def init():
    line.set_data([], [])
    return (line,)

def animate(i):
    
    state = s_a_history[i][0]
    x = (state % 4)+0.5
    y = 3.5 - int(state / 4)
    line.set_data(x, y)
    return (line,)

def simple_convert_into_pi_from_theta(theta):
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    for i in range(0,m):
        pi[i, :] = theta[i, :] / np.nansum(theta[i, :])
    pi = np.nan_to_num(pi)
    
    return pi

def softmax_convert_into_pi_from_theta(theta):
    
    beta = 1.0
    [m, n] = theta.shape
    pi = np.zeros((m, n))
    
    exp_theta = np.exp(beta * theta)
    
    for i in range(0,m):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
    pi = np.nan_to_num(pi)
    
    return pi

def get_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    
    next_direction = np.random.choice(direction, p = pi[s, :])
    
    if next_direction == "up":
        s_next = s - 4
    elif next_direction == "down":
        s_next = s + 4
    elif next_direction == "right":
        s_next = s + 1
    elif next_direction == "left":
        s_next = s - 1
        
    return s_next

def get_action_and_next_s(pi, s):
    direction = ["up", "right", "down", "left"]
    
    next_direction = np.random.choice(direction, p = pi[s, :])
    
    if next_direction == "up":
        action = 0
        s_next = s - 4
    elif next_direction == "down":
        action = 2
        s_next = s + 4
    elif next_direction == "right":
        action = 1
        s_next = s + 1
    elif next_direction == "left":
        action = 3
        s_next = s - 1
        
    return [action, s_next]

def goal_maze(pi):
    s = 0
    s_a_history = [[0, np.nan]]
    while (1):
        [action, next_s] = get_action_and_next_s(pi,s)
        
        s_a_history[-1][1] = action
        s_a_history.append([next_s, np.nan])
        
        
        if next_s == 15:
            break
        else:
            s = next_s
            
    return s_a_history

def update_theta(theta, pi, s_a_history):
    eta = 0.1
    T = len(s_a_history)-1
    
    [m,n] = theta.shape
    delta_theta = theta.copy()
    
    for i in range(0,m):
        for j in range(0,n):
            if not(np.isnan(theta[i, j])):
                SA_i = [SA for SA in s_a_history if SA[0] == i]
                
                SA_ij = [SA for SA in s_a_history if SA == [i,j]]
                
                N_i = len(SA_i)
                N_ij = len(SA_ij)
                
                delta_theta[i,j] = (N_ij - pi[i, j]*N_i) / T
    new_theta = theta + eta * delta_theta
    return new_theta
    
    
    
fig = plt.figure(figsize = (5,5))
ax = plt.gca()

plt.plot([1,1], [0,1], color='blue', linewidth=2)
plt.plot([1,2], [2,2], color='red', linewidth=2)
plt.plot([2,2], [2,1], color='red', linewidth=2)
plt.plot([2,3], [3,3], color='red', linewidth=2)
plt.plot([3,3], [1,3], color='red', linewidth=2)
plt.plot([3,4], [1,1], color='red', linewidth=2)


plt.text(0.5, 3.5, 'S0', size=14, ha='center')
plt.text(1.5, 3.5, 'S1', size=14, ha='center')
plt.text(2.5, 3.5, 'S2', size=14, ha='center')
plt.text(3.5, 3.5, 'S3', size=14, ha='center')
plt.text(0.5, 2.5, 'S4', size=14, ha='center')
plt.text(1.5, 2.5, 'S5', size=14, ha='center')
plt.text(2.5, 2.5, 'S6', size=14, ha='center')
plt.text(3.5, 2.5, 'S7', size=14, ha='center')
plt.text(0.5, 1.5, 'S8', size=14, ha='center')
plt.text(1.5, 1.5, 'S9', size=14, ha='center')
plt.text(2.5, 1.5, 'S10', size=14, ha='center')
plt.text(3.5, 1.5, 'S11', size=14, ha='center')
plt.text(0.5, 0.5, 'S12', size=14, ha='center')
plt.text(1.5, 0.5, 'S13', size=14, ha='center')
plt.text(2.5, 0.5, 'S14', size=14, ha='center')
plt.text(3.5, 0.5, 'S15', size=14, ha='center')
plt.text(0.5, 3.3, 'START', ha='center')
plt.text(3.5, 0.3, 'GOAL', ha='center')

ax.set_xlim(0, 4)
ax.set_ylim(0, 4)
plt.tick_params(axis='both', which='both', bottom='off', top='off', 
                labelbottom='off', right='off', left='off', labelleft='off')

line, = ax.plot([0.5],[3.5], marker='o', color='g', markersize=60)

theta_0 = np.array([[np.nan, 1, 1, np.nan], #S0
                    [np.nan, 1, 1, 1], #S1
                    [np.nan, 1, np.nan, 1], #S2
                    [np.nan, np.nan, 1, 1], #S3
                    [1, 1, 1, np.nan], #S4
                    [1, 1, np.nan, 1], #S5
                    [np.nan, np.nan, 1, 1], #S6
                    [1, np.nan, 1, np.nan], #S7
                    [1, 1, 1, np.nan], #S8
                    [np.nan, np.nan, 1, 1], #S9
                    [1, np.nan, 1, np.nan], #S10
                    [1, np.nan, np.nan, np.nan], #S11
                    [1, np.nan, np.nan, np.nan], #S12
                    [1, 1, np.nan, np.nan], #S13
                    [1, 1, np.nan, 1] #S14
                    ])
    
pi_0 = softmax_convert_into_pi_from_theta(theta_0)


s_a_history = goal_maze(pi_0)

stop_epslion = 10**-4

theta = theta_0
pi = pi_0

is_continue = True
count = 1
while is_continue:
    s_a_history = goal_maze(pi)
    new_theta = update_theta(theta, pi, s_a_history)
    new_pi = softmax_convert_into_pi_from_theta(new_theta)
    
    print(np.sum(np.abs(new_pi-pi)))
    print("走出迷宮的總步數為" + str(len(s_a_history) -1) + "呦")
    
    if np.sum(np.abs(new_pi - pi)) < stop_epslion:
        is_continue = False
    else:
        theta = new_theta
        pi = new_pi
    count += 1
np.set_printoptions(precision=3, suppress=True)
print(pi)
#print(s_a_history)
#print("走出迷宮的總步數為" + str(len(s_a_history)) + "呦")

anim = animation.FuncAnimation(fig, animate, init_func = init, frames = len(s_a_history), 
                               interval = 200, repeat = False)
HTML(anim.to_jshtml())
import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from tqdm import tqdm
import matplotlib.pyplot as plt


# Intializing positions. Positions of agents are ranomly set within the square arena
# position array also stores each agent's interaction radius, intrinsic speed and intrinsic noise
def init_position(position,num_1,num_2,num_part,box_length,rad1,rad2,v_abs1,v_abs2,noise1,noise2):
    
    position[:, 0] = np.random.uniform(0, box_length, num_part)
    position[:, 1] = np.random.uniform(0, box_length, num_part) 
        
    position[0:num_1,2] = rad1 
    position[num_1:num_part,2] = rad2
    position[0:num_1,3] = v_abs1
    position[num_1:num_part,3] = v_abs2
    position[0:num_1,4] = noise1
    position[num_1:num_part,4] = noise2
    return(position)
 
# Initializing velocities. Orientations of each agent are randomly selected and corresponding velocities
# are calculated using intrinsic speed 
def init_velocity(num_part,box_length,angles,velocity,v_abs1,v_abs2,num_1,num_2):

   angles[:,0] = np.random.uniform(-np.pi, np.pi, num_part)

   velocity[0:num_1] = v_abs1*np.cos(angles[0:num_1])
   velocity[num_1:num_part] = v_abs2*np.cos(angles[num_1:num_part])



def velocity_update(velocity,angles,num_part,box_length,position):
    
    x_avg = np.empty([num_part])
    y_avg = np.empty([num_part])
    mod = np.empty([num_part])
    for i in range(num_part):
        
        # Accounting for periodic boundaries
        x_dist = np.abs(position[:, 0] - position[i, 0])
        y_dist = np.abs(position[:, 1] - position[i, 1])
        
        x_dist = np.minimum(x_dist, box_length - x_dist)
        y_dist = np.minimum(y_dist, box_length - y_dist)
        
        # Getting truth table of neighbours list; 
        # if agents are within the interaction radius of focal agent, that index is set to True
        neighbour_list = (x_dist)**2 + (y_dist)**2 <= position[i,2]**2
        
        # Taking average of x and y vectors to get average direction the agent must face
        x_avg[i] = (np.sum(np.cos(angles[neighbour_list])))/np.count_nonzero(neighbour_list == True)
        y_avg[i] = (np.sum(np.sin(angles[neighbour_list])))/np.count_nonzero(neighbour_list == True)
        
        # Making it a unit vector
        mod[i] = np.sqrt(x_avg[i]**2 + y_avg[i]**2)
        x_avg[i]/= mod[i]
        y_avg[i]/= mod[i]
    
    
    # updating new velocities(direction since speeds are fixed)         
    for j in range(num_part): 
        # adding random noise to updated orientation         
        angles[j] = np.arctan2(y_avg[j],x_avg[j]) + np.random.uniform(-position[j,4]*np.pi,position[j,4]*np.pi)
    
    
        velocity[j,0] = position[j,3]*np.cos(angles[j])
        velocity[j,1] = position[j,3]*np.sin(angles[j])
        
    
        


def position_update(num_particles,position,velocity,box_length,dt):

    l = box_length
    dt = np.sqrt(dt)
    

    for i in range(num_particles):
        position[i,0]+= velocity[i,0]*dt 
        position[i,1]+= velocity[i,1]*dt 
        
        # Periodic boundaries
        if position[i,0] > l: position[i,0] = math.fmod(position[i,0],l)
        if position[i,0] < 0: position[i,0] = math.fmod(position[i,0],l) + l
      
        if position[i,1] > l: position[i,1] = math.fmod(position[i,1],l)
        if position[i,1] < 0: position[i,1] = math.fmod(position[i,1],l) + l

    return(position)



# Runs model to get PCA components
# Also prints position data in .xyz file format to animate in OVITO
def run_model(position,angles,velocity,num_1,num_2,num_particles,box_length,dt,rad1,rad2,v_abs1,v_abs2,noise1,noise2,num_iter,cutoff_time,X_t):
    
    init_position(position,num_1,num_2,num_particles,box_length,rad1,rad2,v_abs1,v_abs2,noise1,noise2)
    init_velocity(num_particles,box_length,angles,velocity,v_abs1,v_abs2,num_1,num_2)
    
    data = []
    
    
    for i in tqdm(range(num_iter)):
        
        
        velocity_update(velocity,angles,num_particles,box_length,position)
        position_update(num_particles,position,velocity,box_length,dt)

        
        if i >=cutoff_time:
            
            # Generation of data matrix for PCA, here we are storing sin and cos values of angle 
            X_t[0:num_particles,i - cutoff_time] = np.cos(angles[:,0])
            X_t[0:num_particles,i + (num_iter - cutoff_time) - cutoff_time] = np.sin(angles[:,0])
            
        
        snapshot = [f"{num_particles}\nParticles\n"]
        snapshot.extend([f"1 {position[j, 0]} {position[j, 1]} 0\n" for j in range(num_1)])
        snapshot.extend([f"2 {position[j, 0]} {position[j, 1]} 0\n" for j in range(num_1, num_particles)])
        data.append("".join(snapshot))
   
    
    with open("vicsek_test.xyz", "w") as f:
        f.write("".join(data))
    
        
     
    # Processing of data matrix to obtain PCs
    typeone = ['typeone' + str(i) for i in range(1,num_1+1)]
    typetwo = ['typetwo' + str(i-num_1) for i in range(num_1+1,num_particles+1)]
    scaled_X_t = preprocessing.scale(X_t)
    pca = PCA()
    pca.fit(scaled_X_t)
    pca_data = pca.transform(scaled_X_t)
    
    per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
    labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]
    
    # Stores all PCs
    pca_df = pd.DataFrame(pca_data,index=[*typeone,*typetwo],columns=labels)
    return(pca_df)



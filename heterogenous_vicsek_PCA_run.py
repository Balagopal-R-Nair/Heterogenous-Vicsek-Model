import numpy as np
import matplotlib.pyplot as plt


from  heterogenous_vicsek_PCA_functions import run_model




#######################################
############# PARAMETERS #############
#######################################

box_length = 1                          # Length of square arena
num_1 = 200                             # Number of agents of type 1
num_2 = 20                              # Number of agents of type 2
num_particles = num_1 + num_2           # Total number of agents
v_abs1 = 0.01                           # Intrinsic speed of type 1 agents
v_abs2 = 0.01                           # Intrinsic speed of type 2 agents
noise1 = 0.1                            # Intrinsic noise of type 1 agents
noise2 = 0.4                            # Intrinsic noise of type 2 agents
rad1 = 0.05                             # Interaction radius of type 1 agents
rad2 = 0.05                             # Interaction radius of type 2 agents
dt = 0.1                                # size of time step
num_iter = 9000                         # Number of time steps
cutoff_time = 1000                      # Cutoff time after which data is stored


X_t = np.empty([num_particles,2*(num_iter - cutoff_time)])  # Data matrix on which PCA is done

# Stores corresponding quantities for a time step
position = np.empty([num_particles,5])  # also stores information on intrinsic properties of each type of agent
angles =  np.empty([num_particles,1])
velocity = np.empty([num_particles,2])

#######################################
#######################################
#######################################


pca_df = run_model(position,angles,velocity,num_1,num_2,num_particles,box_length,dt,rad1,rad2,v_abs1,v_abs2,noise1,noise2,num_iter,cutoff_time,X_t)




# Plotting PC1 vs PC2
plt.scatter(pca_df.PC1[0:num_1],pca_df.PC2[0:num_1], label = 'type 1')
plt.scatter(pca_df.PC1[num_1:num_particles],pca_df.PC2[num_1:num_particles], label = 'type 2')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.legend()
plt.show()
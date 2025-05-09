# %%

import numpy as np
import json

class ChemicalNetwork:
    def __init__(self, num_species=None, num_input=None, num_output=None, \
                 n_desc=None, conc_hidden=None, conc_out=None, \
                    config_file=None):    
        """
        Initialize the Chemical Network

        :param num_species: Number of chemical species in the network
        :n_desc: Number of descendants of K
        :conc_hidden/out: concentration of the species in the hidden/out layers
        """
        if config_file:
            self.load_config(config_file)
        else:
            self.num_species = num_species
            self.num_input = num_input      # number of input nodes
            self.num_output = num_output    # number of output nodes
            self.num_desc = n_desc 
            self.conc_hidden = conc_hidden
            self.conc_out = conc_out
        self.conc_hidden_min = 0.2
        self.conc_hidden_max = 5.0

        if (self.num_input + self.num_output > self.num_species):  
             raise ValueError("Input and output layers overlap")

        self.concentration = np.ones(self.num_species) # total concentrations
        self.concentration[:self.num_species-self.num_output] = self.conc_hidden # the input concentrations are overwritten when running inference dynamics 
        self.concentration[self.num_species-self.num_output:] = self.conc_out
        self.list_conc_out = self.conc_out*np.ones(self.num_desc)
        self.list_conc_hidden = self.conc_hidden*np.ones(self.num_desc)

        # initialisation of the dynamic variables (affinities)
        self.states = [np.zeros(self.num_species) for _ in range(self.num_desc)] # neuronal variables for all descendants

        # interaction matrix 
        self.Ks = [np.zeros((self.num_species, self.num_species)) for _ \
                   in range(self.num_desc)]
        self.Ksymm = np.ones((self.num_species, self.num_species)) # used to enforce symmetry of Ks 
        for i in range(1,self.num_species):
            for j in range(0,i):
                self.Ksymm[i,j]=0.0

        # print('Ksymm=',self.Ksymm)
        self.Kmask = np.ones((self.num_species, self.num_species))
        for i in range(0,self.num_species): # used to set self-interactions = 0
            self.Kmask[i,i] = 0.0
        self.Kmask[:self.num_input,:self.num_input] = 0.0 # used to switch off the interaction between input species 
        # print('Kmask=',self.Kmask)

        # score ranks the different models 
        self.scores = np.zeros(self.num_desc) 

    def load_config(self, config_file):
        """
        Load configuration from a JSON file and initialize the class attributes.
        :param config_file: Path to the configuration file (JSON format).
        """
        with open(config_file, 'r') as f:
            config = json.load(f)
            self.num_species = config['num_species']
            self.num_input = config['num_input']
            self.num_output = config['num_output']
            self.num_desc = config['num_desc']
            self.conc_hidden = config['conc_hidden']
            self.conc_out = config['conc_out']

    def save_config(self, config_file):
        """
        Save the current configuration to a JSON file.
        :param config_file: Path to the configuration file (JSON format).
        """
        config = {
            'num_species': self.num_species,
            'num_input': self.num_input,
            'num_output': self.num_output,
            'num_desc': self.num_desc,
            'conc_hidden': self.conc_hidden,
            'conc_out': self.conc_out
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4) 

    def load_k_matrix(self, k_file):
        """
        Load the K matrix from a JSON file.
        :param k_file: Path to the K matrix file (JSON format).
        """
        with open(k_file, 'r') as file:
            k_data = json.load(file)
            self.Ks[0] = np.array(k_data['K'])

    def save_k_matrix(self, k_file):
        K_list = self.Ks[0].tolist()
        k_data = {
            "K": K_list
        }
        with open(k_file, 'w') as file:
            json.dump(k_data, file, indent=4)

    def clone_K(self,mean,sigma): # this routine should be vectorized 
        K=self.Ks[0]
        for i in range(1,self.num_desc):
            normal_matrix = np.random.lognormal(mean=mean, sigma=sigma, size=(self.num_species,self.num_species))
            normal_matrix = normal_matrix*self.Kmask*self.Ksymm
            normal_matrix = normal_matrix+normal_matrix.T
            self.Ks[i] = np.minimum(K*normal_matrix,np.power(10.,4.))  

    def initialise_K(self,mean,sigma,mean_0=None,sigma_0=None,K_file=None, \
                     ):
        if(K_file):
            self.load_k_matrix(K_file)
        else:
            K= np.random.lognormal(mean=mean_0, sigma=sigma_0, size=(self.num_species,self.num_species))
            K=K*self.Kmask*self.Ksymm
            K=K+K.T
            self.Ks[0]=K
        self.clone_K(mean, sigma)

# vectorized
    def network_dynamics(self, K, Input_concentrations, conc_hidden, conc_out, precision):
        """
        Vectorized network dynamics where all patterns share the same interaction matrix (K).

        :param K: Interaction matrix (shape: [num_species, num_species])
        :param Input_concentrations: Batch of input concentration vectors (shape: [batch_size, num_input])
        :param conc_hidden: Hidden layer concentration
        :param conc_out: Output layer concentration
        :param precision: Tolerance for convergence
        :return: Final states for all input patterns, and the number of iterations per pattern
        """
        batch_size = Input_concentrations.shape[0]  # Number of input patterns
        num_species = self.num_species

        # Initialize concentrations for all input patterns
        concentrations = np.ones((batch_size, num_species)) * conc_hidden
        concentrations[:, :self.num_input] = Input_concentrations  # Set input concentrations
        concentrations[:, -self.num_output:] = conc_out  # Set output concentrations

        # Initialize states and convergence variables
        states = np.zeros((batch_size, num_species))  # Shape: [batch_size, num_species]
        prev_states = np.zeros_like(states)
        ac_precision = 2.0 * precision * np.ones(batch_size)  # One precision value per pattern
        # max_iterations = 1000  # Optional: Set a maximum number of iterations
        iteration_counts = np.zeros(batch_size, dtype=int)  # Track iterations for each pattern

        # Iteratively solve for equilibrium concentrations
        while np.any(ac_precision > precision):  # Continue until all patterns converge
            prev_states = states.copy()
            states = concentrations / (1.0 + np.dot(states, K.T))  # Update states in parallel for all patterns
            deltas = 2.0 * np.abs(prev_states - states) / (np.abs(prev_states + states) + precision / 10)
            ac_precision = np.max(deltas, axis=1)  # Update precision for each pattern
            iteration_counts += 1
#        if np.all(iteration_counts >= max_iterations):  # Safety check to prevent infinite loops
#            break

        return states, iteration_counts


    def compute_losses(self,labels,outputs,loss,out_losses):
        N_patt = labels.shape[0]
        c_inf = np.min(labels)
        c_sup = np.max(labels)

        if loss == 1:  # entropic loss

            s_matrix = (labels - c_inf) / (c_sup - c_inf) 
            outputs_clipped = np.clip(outputs, 1e-10, self.conc_out)
            off_list=np.log(np.max((1-s_matrix)*outputs_clipped,axis=1))
            on_list=np.min(s_matrix*np.log(outputs_clipped),axis=1)
            list_losses=off_list-on_list
            score = np.average(list_losses)

            ## two old versions 
            #indinc = (labels - c_inf) / (c_sup - c_inf)  # Shape: [N_patt, num_output]
            #outputs_clipped = np.clip(outputs, 1e-10, self.conc_out)  # to avoid computing 1./0. and log(0.)  
            #num = np.sum(indic * outputs, axis=1)            
            #den = np.max((1-indic) * outputs, axis=1)
            ##if out_losses == 1:
            #list_losses = -np.log(num/den)    
            #score = np.sum(-np.log(num/den))
            ##
            ## version with preaverage over the patterns belonging to the same class  
            ##
            #label_indices = np.argmax(labels, axis=1) # Shape: [N_patt]; label_indices=0,1,2
            #num_labels=labels.shape[1]
            #score_label=np.zeros(num_labels)
            #for label in range(num_labels):
            #    mask = (label_indices == label)  # Boolean mask for current label    
            #    score_label[label]=np.sum(list_losses[mask])
            #score=np.max(score_label)

        elif loss == 2:  # mean squared error (mse) 

            if out_losses == 1:
                list_losses = np.sum(np.power(outputs - labels, 2),axis=1)
            score = np.average(np.power(outputs - labels, 2))*3./2

        elif loss == 3: # contrast loss

            s_matrix = (labels - c_inf) / (c_sup - c_inf)  # Shape: [N_patt, num_output]
            outputs_clipped = np.clip(outputs, 1e-10, self.conc_out)/self.conc_out  # regularizators  

            #label_indices = np.argmax(labels, axis=1) # Shape: [N_patt]; label_indices=0,1,2
            #num_labels=labels.shape[1]
            on_matrix=s_matrix*np.log(outputs_clipped)            
            off_matrix=(1.-s_matrix)*outputs_clipped

            ## first version of the score 
            # score=-np.average(on_matrix)+0.5*np.average(np.log(np.average(off_matrix,axis=0)))
            list_scores=np.log(3./2.*np.average(off_matrix,axis=0))-3.*np.average(on_matrix,axis=0)
            score=np.max(list_scores)
            
            if out_losses == 1:
                list_losses=np.average(on_matrix,axis=1)

        elif loss == 4: # symmetric loss
            
            s_matrix = (labels - c_inf) / (c_sup - c_inf)  # Shape: [N_patt, num_output]
            outputs_clipped = np.clip(outputs, 1e-10, self.conc_out)/self.conc_out  # regularizators 

            on_matrix=np.log(1-np.min(s_matrix*outputs_clipped,axis=1))
            off_matrix=np.log(np.max((1-s_matrix)*outputs_clipped,axis=1))

            list_losses=on_matrix+off_matrix
            score=np.average(list_losses)


        else:
            raise ValueError("Invalid loss type. Must be 1, or 2, or 3, or 4.")

        if out_losses == 1:

            return score , list_losses
        else:
            return score 

# vectorized 
    def test_models(self, patterns, labels, K, conc_hidden, conc_out, precision, loss, out_losses):
        """
        Test models: We assume a one-hot encoding of the labels which are assumed to be equal to (c_inf, c_inf, ..., c_sup, c_inf, ...)

        :param loss: 
            0 - Cross entropy loss
            1 - Unconventional loss (requires c_inf=0 and c_sup=c_out)
            2 - Mean squared (MS) loss
        :return: Average loss across all patterns
        """
#        N_patt = patterns.shape[0]  # Number of patterns
        i_out = self.num_species - labels.shape[1]  # Output indices
#        c_inf = np.min(labels)
#        c_sup = np.max(labels)

        # Run vectorized network dynamics for all patterns
        states, _ = self.network_dynamics(K, patterns, conc_hidden, conc_out, precision)

        # Extract output states for all patterns
        outputs = states[:, i_out:]  # Shape: [N_patt, num_output]
        
        if out_losses == 1:
            score , list_losses = self.compute_losses(labels,outputs,loss,out_losses)
            return score , list_losses
        else:
            score = self.compute_losses(labels,outputs,loss,out_losses)
            return score 


    def parallel_test_models(self, patterns, labels, conc_hidden, conc_out, precision, loss, out_losses):  # Now fully sequential

        n_des = len(self.Ks)
        n_batch=patterns.shape[0]
        self.scores = []  # Clear the previous scores

        if out_losses == 1:
            tab_losses = np.empty((0,n_batch))

        for i in range(n_des):  # Sequential iteration over all Ks while probing different concentrations
            score_list=np.zeros(3)
            concs_hidden=conc_hidden*np.array([0.985,1.0,1.0/0.985])
            concs_hidden=np.minimum(concs_hidden,self.conc_hidden_max)
            concs_hidden=np.maximum(concs_hidden,self.conc_hidden_min)
            if out_losses == 1:
                score_list[0], losses_0 = self.test_models(patterns, labels, self.Ks[i], concs_hidden[0], conc_out, precision,loss, out_losses)
                score_list[1], losses_1 = self.test_models(patterns, labels, self.Ks[i], concs_hidden[1], conc_out, precision,loss, out_losses)
                score_list[2], losses_2 = self.test_models(patterns, labels, self.Ks[i], concs_hidden[2], conc_out, precision,loss, out_losses)
            else:
                score_list[0] = self.test_models(patterns, labels, self.Ks[i], concs_hidden[0], conc_out, precision,loss, out_losses)
                score_list[1] = self.test_models(patterns, labels, self.Ks[i], concs_hidden[1], conc_out, precision,loss, out_losses)
                score_list[2] = self.test_models(patterns, labels, self.Ks[i], concs_hidden[2], conc_out, precision,loss, out_losses)
            iopt=np.argmin(score_list)
            self.list_conc_hidden[i]=concs_hidden[iopt]
            self.scores.append(score_list[iopt])  # Store the resulting score
            if out_losses == 1:
                if iopt == 0:
                    losses=losses_0
                elif iopt == 1:
                    losses=losses_1
                elif iopt == 2:
                    losses=losses_2
                tab_losses = np.vstack(([tab_losses,losses]))        
        if out_losses == 1:
            return tab_losses


    def update_K(self,mean,sigma):
        iopt=np.argmin(self.scores) # carefull, this argmax will pick up the first index when the scores are equal
        self.scores[0]=self.scores[iopt]
        self.conc_hidden=self.list_conc_hidden[iopt]
        self.Ks[0]=np.copy(self.Ks[iopt])
        self.clone_K(mean,sigma)
        self.list_conc_hidden=self.conc_hidden*np.ones(self.num_desc)


    # used to print out average and variance versus time of the output activities 
    # vectorized
    def output_activities(self, patterns, labels, conc_hidden, conc_out, precision, loss):
        """
        Compute affinities of the most performant model, assuming a one-hot encoding of labels.

        :param patterns: Array of input patterns (shape: [N_patt, num_input])
        :param labels: Array of one-hot encoded labels (shape: [N_patt, num_output])
        :param conc_hidden: Hidden layer concentration
        :param conc_out: Output layer concentration
        :param precision: Precision for equilibrium calculation
        :return: Normalized activities (shape: [num_labels, num_output])
        """
        N_patt = patterns.shape[0]  # Number of patterns
        i_out = self.num_species - labels.shape[1]  # Output indices
        num_output = self.num_species - i_out       # Number of output species
        num_labels = labels.shape[1]               # Number of unique labels

        # Initialize mean activities, variance of the activities, and counts
        activities = np.zeros((num_labels, num_output))
        variance_activity = np.zeros((num_labels, num_output))
        #counts = np.zeros((num_labels, num_output))

        # Run network dynamics for all patterns in a batch
        states, _ = self.network_dynamics(self.Ks[0], patterns, conc_hidden, conc_out, precision)

        # Extract output states for all patterns
        outputs = states[:, i_out:]  # Shape: [N_patt, num_output]

        # Determine the label indices for each pattern (assuming one-hot encoding)
        label_indices = np.argmax(labels, axis=1)  # Shape: [N_patt]

        # Vectorized aggregation of activities and counts
        for label in range(num_labels):
            mask = (label_indices == label)  # Boolean mask for current label
            activities[label] = np.mean(outputs[mask], axis=0) 
            variance_activity[label] = np.var(outputs[mask], axis=0)
        #    counts[label] = np.sum(mask)  # Count occurrences of this label

# no divisions by zero when using balanced batches (i.e., with the same number of patterns/class)             
#        # Normalize activities by counts
#        with np.errstate(divide='ignore', invalid='ignore'):  # Handle division by zero safely
#            # normalized_activities = np.nan_to_num(activities / counts[:, None])
#            normalized_activities = np.nan_to_num(activities / counts)

        out_losses=0
        ac_score = self.compute_losses(labels,outputs,loss,out_losses)
        
        return ac_score, activities, variance_activity

# vectorized
    def test_model(self, patterns, labels, conc_hidden, conc_out, precision):
        """
        Prepares scatter plot data by computing output states and grouping by label.

        :param patterns: Array of input patterns (shape: [N_patt, num_input])
        :param labels: Array of one-hot encoded labels (shape: [N_patt, num_labels])
        :param conc_hidden: Hidden layer concentration
        :param conc_out: Output layer concentration
        :param precision: Precision for equilibrium calculation
        :return: Arrays of affinities grouped by label (one per label)
        """
        # Output layer size and number of labels
        i_out = self.num_species - labels.shape[1]
        num_output = self.num_species - i_out
        num_labels = labels.shape[1]

        # Run network dynamics for all patterns
        states, _ = self.network_dynamics(self.Ks[0], patterns, conc_hidden, conc_out, precision)

        # Extract outputs (equilibrium states of the output species)
        outputs = states[:, i_out:]  # Shape: [N_patt, num_output]

        # Determine label indices for each pattern (assuming one-hot encoding)
        label_indices = np.argmax(labels, axis=1)  # Shape: [N_patt]

        # Initialize affinities for each label as lists
        affinities = [np.empty((0, num_output)) for _ in range(num_labels)]

        # Group outputs by their label using vectorized processing
        for label in range(num_labels):
            mask = (label_indices == label)  # Boolean mask for current label
            affinities[label] = outputs[mask]  # Collect outputs for this label

        return affinities



# %% [markdown]
# Mutual information calculation 

# %%
def compute_entropy(x, dx=0.02):
    """
    Compute the discrete Shannon entropy for a one-dimensional array x.
    The distribution is estimated using a histogram with bin width dx.
    
    Parameters:
      x  : 1D numpy array of data.
      dx : The width of each bin (default 0.02).
    
    Returns:
      The computed Shannon entropy.
    """
    # Define bin edges from the minimum to maximum value of x, using dx as step.
    edges = np.arange(np.min(x), np.max(x) + dx, dx)
    counts, _ = np.histogram(x, bins=edges, density=False)
    total = np.sum(counts)
    if total == 0:
        return 0
    p = counts / total
    # Filter out zero probabilities to avoid log(0)
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def compute_mutual_information(a_list):
    Dec = 1 / np.log(10)
    I_max=1/3*np.log(3)+2/3*np.log(3/2)
    # Compute transformed data for each dataset.
    all_x, all_y, all_z = [], [], []
    for a in a_list:
        x = Dec * np.log(a[:, 0])
        y = Dec * np.log(a[:, 1])
        z = Dec * np.log(a[:, 2])
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)

    H0 = compute_entropy(all_x[0])
    H1 = compute_entropy(all_y[1])
    H2 = compute_entropy(all_z[2])
    combined_12 = np.concatenate([all_x[1], all_x[2]])
    combined_02 = np.concatenate([all_y[0], all_y[2]])
    combined_01 = np.concatenate([all_z[0], all_z[1]])
        # Combined: concatenate all x values.
    combined_x = np.concatenate(all_x)
    combined_y = np.concatenate(all_y)
    combined_z = np.concatenate(all_z)
    H_12 = compute_entropy(combined_12)
    H_02 = compute_entropy(combined_02)
    H_01 = compute_entropy(combined_01)


    mutual_information_x = compute_entropy(combined_x) - (1/3 * H0 + 2/3 * H_12)
    mutual_information_y = compute_entropy(combined_y) - (1/3 * H1 + 2/3 * H_02)
    mutual_information_z = compute_entropy(combined_z) - (1/3 * H2 + 2/3 * H_01)
    mutual_information_x=mutual_information_x/I_max
    mutual_information_y=mutual_information_y/I_max
    mutual_information_z=mutual_information_z/I_max

    return mutual_information_x, mutual_information_y, mutual_information_z



# %%

# input files (someone may be None by default)
K_init_file=None # 'K_matrix_GD.json' # None if we want to initialize K from scratch
network_config_file='network_config_EL.json'
patterns_run_config_file='patterns_run_config_EL.json'
patterns_0_file='patterns_0.npy' # None if we want to initialize patterns from scratch

with open(patterns_run_config_file, 'r') as f:
    config = json.load(f)
    sigma2_0 = config['sigma2_0']
    mean_0 = config['mean_0']
    mean = config['mean']
    sigma = config['sigma']
    sigma2 = config['sigma2']
    loss = config['loss']
    unpert_pattern = config['unpert_pattern']
    precision = config['precision']
    N_batches_passed= config['N_batches_passed']
    N_patt= config['N_patt']

sigma_0=np.sqrt(sigma2_0)
D=np.sqrt(sigma2) # variance of the noise 

# output file 
#patterns_0_file=f"patterns_0_{unpert_pattern}.npy" 

#scatter_unlearned_file=f"a1a2a3_NotLed_{unpert_pattern}_D2{sigma2}.npz"  
#scatter_file=f"a1a2a3_{unpert_pattern}_D2{sigma2}.npz"
#loss_file=f"loss_{unpert_pattern}_D2{sigma2}.npz"
#activities_file=f"activities_{unpert_pattern}_D2{sigma2}.npz"
#hidden_file=f"hidden_conc_{unpert_pattern}_D2{sigma2}.npz"
#K_fin_matrix=f"K_matrix_{unpert_pattern}_D2{sigma2}.json"
#CN_state_file=f"CN_state_{unpert_pattern}_D2{sigma2}.npz"
#list_K_file=f"list_K_matrix_{unpert_pattern}_D2{sigma2}.npy"

scatter_unlearned_file=f"a1a2a3_NotLed_D2.npz" 
scatter_file=f"a1a2a3_D2.npz"
loss_file=f"loss_D2.npz"
activities_file=f"activities_D2.npz"
hidden_file=f"hidden_conc_D2.npz"
K_fin_matrix=f"K_matrix_D2.json"
CN_state_file=f"CN_state_D2.npz"
list_K_file=f"list_K_matrix_D2.npy"

CN=ChemicalNetwork(num_species=None, num_input=None, num_output=None, \
                 n_desc=None, conc_hidden=None, conc_out=None, \
                    config_file=network_config_file)

#CN=ChemicalNetwork(N_species,N_inp,N_out,N_desc,Conc_hidden,Conc_out)


if K_init_file is None:
    CN.initialise_K(mean,sigma, mean_0=mean_0,sigma_0=sigma_0,K_file=None)
else:
    CN.initialise_K(mean,sigma, mean_0=None, sigma_0=None, K_file=K_init_file)

if(unpert_pattern==None):
    patterns_0=np.random.lognormal(mean=0, sigma=np.sqrt(2.0),size=(3,6))              
else:
    patterns_0=np.load(patterns_0_file)
    
#if(unpert_pattern=='A'):
#    patterns_0=np.array([[4.9 , 0.54 , 0.78 ,  0.11 ,  0.70 ,  1.5 ], 
#     [3.9 ,  0.37 ,  0.33 ,  0.95 , 0.60 ,  1.4 ], 
#     [3.1 ,  0.80 ,  0.07 ,  1.1 ,   0.18 , 26.8 ]])

#if(unpert_pattern=='B'):
#    patterns_0=np.array([[ 0.24001358,  3.19728681,  9.23301249,  1.44793741,  0.79277543,  0.56445251],
#     [ 0.28378802,  1.98128614,  4.97609186,  0.19833512,  2.35384955,  0.09018742],
#     [ 1.55839014,  0.52222897,  0.913423,    7.07690945, 41.08854987,  0.5581125 ]])

#if(unpert_pattern=='C'):
#    patterns_0=np.array([[ 1.32841165,  0.38709303,  2.29672817,  9.45836126, 20.61518202,  0.09458833],
# [33.60512186,  0.7034888,   2.28065926,  3.14203644,  5.32061224,  0.46844228],
# [ 0.58673294,  1.16767597,  8.43878063,  0.10142915,  0.69748406,  0.89055241]])

#if(unpert_pattern=='D'):
#    patterns_0=np.array([[0.57127289, 0.83179442, 2.14768857, 5.98053458, 0.69395479, 3.37960999],
# [2.27496627, 3.21611655, 7.33597589, 0.49539532, 3.31741288, 0.3586194 ],
# [3.83556362, 5.05860257, 2.5699809,  1.84195934, 0.19433368, 1.49518432]])

labels_0 = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])     
dim_pattern=np.shape(patterns_0)[1]

def gen_patt_label(N_patt, D):
    # Randomly select indices from patterns_0
    # indices = np.random.randint(0, patterns_0.shape[0], size=N_patt)    
    # Balanced number of patterns 
    indices=np.zeros(N_patt,dtype=int)
    N1=int(N_patt/3)
    N2=2*N1
    indices[N1:]+=1
    indices[N2:]+=1
    selected_labels = labels_0[indices, :]
    selected_patterns = patterns_0[indices, :]
    # Apply log-normal scaling to the patterns
    scaling_factors = np.random.lognormal(mean=0, sigma=D, size=(N_patt, selected_patterns.shape[1]))
    scaled_patterns = selected_patterns * scaling_factors
    return scaled_patterns, selected_labels


np.save(patterns_0_file, patterns_0)



#training 

list_batch_pass = np.array([])
list_score = np.array([])
list_conc_hidden = np.array([])
list_Ks=[]
list_a_patt1 = np.empty((0,np.shape(labels_0)[1]))
list_a_patt2 = np.empty((0,np.shape(labels_0)[1]))
list_a_patt3 = np.empty((0,np.shape(labels_0)[1]))
list_a_var_patt1 = np.empty((0,np.shape(labels_0)[1]))
list_a_var_patt2 = np.empty((0,np.shape(labels_0)[1]))
list_a_var_patt3 = np.empty((0,np.shape(labels_0)[1]))
list_table_losses = []


# scatter plot for the untrained model 
patterns_scatter, labels_scatter = gen_patt_label(5*N_patt,D)
a1_0, a2_0, a3_0 = CN.test_model(patterns_scatter, labels_scatter, CN.conc_hidden, \
                                 CN.conc_out, precision)
np.savez(scatter_unlearned_file, arr1=a1_0, arr2=a2_0,arr3=a3_0)

# training 
i_batch=0
while(i_batch<N_batches_passed):

    patterns, labels = gen_patt_label(N_patt,D)
    Conc_hidden = CN.conc_hidden
    Conc_out = CN.conc_out

    if(np.mod(10*i_batch,N_batches_passed)==0):
        out_loss=1
        tl=CN.parallel_test_models(patterns,labels,Conc_hidden,Conc_out,precision,loss,out_loss)
        list_table_losses.append(tl)
    else:
        out_loss=0
        CN.parallel_test_models(patterns,labels,Conc_hidden,Conc_out,precision,loss,out_loss)

    CN.update_K(mean,sigma)
    Conc_hidden = CN.conc_hidden
    Conc_out = CN.conc_out

    # testing the model    
    if(np.mod(i_batch,10) == 0):
        patterns_test, labels_test = gen_patt_label(4*N_patt,D) 
        ac_score, activities, variances = CN.output_activities(patterns_test,labels_test,Conc_hidden,Conc_out,precision,loss)
        list_batch_pass = np.append(list_batch_pass,i_batch)
        list_a_patt1 = np.vstack((list_a_patt1,activities[0]))
        list_a_patt2 = np.vstack((list_a_patt2,activities[1]))
        list_a_patt3 = np.vstack((list_a_patt3,activities[2]))
        list_a_var_patt1 = np.vstack((list_a_var_patt1,variances[0]))
        list_a_var_patt2 = np.vstack((list_a_var_patt2,variances[1]))
        list_a_var_patt3 = np.vstack((list_a_var_patt3,variances[2]))
        list_score = np.append(list_score,ac_score)
        list_conc_hidden = np.append(list_conc_hidden,CN.conc_hidden)

    i_batch = i_batch+1



    if(np.mod(10*i_batch,N_batches_passed)==0):
        list_Ks.append(CN.Ks[0])
        print('i_batch=',i_batch)


a1, a2, a3 = CN.test_model(patterns_scatter, labels_scatter, CN.conc_hidden, \
                                 CN.conc_out, precision)
np.savez(scatter_file, arr1=a1, arr2=a2,arr3=a3)


# %%


# %%
# Save loss function
np.savez(loss_file, arr1=list_batch_pass, arr2=list_score)

# Save the activities evolution
np.savez(activities_file, arr1=list_batch_pass, arr2=list_a_patt1, arr3=list_a_var_patt1, \
                                         arr4=list_a_patt2, arr5=list_a_var_patt2, \
                                         arr6=list_a_patt3, arr7=list_a_var_patt3)


# Save hidden concentration
np.savez(hidden_file, arr1=list_batch_pass, arr2=list_conc_hidden)

# Save the learned affinity matrix K
CN.save_k_matrix(K_fin_matrix)

out=CN.network_dynamics(CN.Ks[0], patterns_0, CN.conc_hidden, CN.conc_out, precision=np.power(10.,-10))[0]
np.savez(CN_state_file, arr1=out[0], arr2=out[1],arr3=out[2])

# Save list_Ks to a file
np.save(list_K_file, list_Ks)



# %%

#list_table_losses = np.array(list_table_losses)  

#filename = f"Results/list_table_losses_{unpert_pattern}_D{D2}.npz"
#np.savez(filename, arr1=list_table_losses)

#it=9

#fig, axes = plt.subplots(1, 3, figsize=(10, 5))  # 1 row, 2 columns

## First panel
#im1 = axes[0].imshow(list_table_losses[it, :, :15], cmap='Blues')
#fig.colorbar(im1, ax=axes[0])  # Add colorbar to the first panel
#axes[0].set_title("pattern 0")
#axes[0].set_xlabel("Batch Index")
#axes[0].set_ylabel("Models")

## Second panel
#im2 = axes[1].imshow(list_table_losses[it, :, 16:31], cmap='Blues')  # Change colormap if needed
#fig.colorbar(im2, ax=axes[1])  # Add colorbar to the second panel
#axes[1].set_title("pattern 2")
#axes[1].set_xlabel("Batch Index")
#axes[1].set_ylabel("Models")

## Second panel
#im3 = axes[2].imshow(list_table_losses[it, :, 32:], cmap='Blues')  # Change colormap if needed
#fig.colorbar(im2, ax=axes[2])  # Add colorbar to the second panel
#axes[2].set_title("pattern 3")
#axes[2].set_xlabel("Batch Index")
#axes[2].set_ylabel("Models")

#plt.tight_layout()  # Adjust layout to prevent overlap
#plt.show()





# %%
            



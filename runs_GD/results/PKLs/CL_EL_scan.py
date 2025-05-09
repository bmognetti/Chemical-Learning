import sys
import numpy as np
import pickle
import os

#index_run=1
index_run = int(sys.argv[1])
drift_scan=[8
            ]
noise_scan=range(10,121,5)


#  ---------------------------------
unpert_pattern=f"F{index_run}"


class ChemicalNetwork:

    def __init__(self, num_species, num_input, num_output, n_desc, conc_hidden, conc_out):
        """
        Initialize the Chemical Network

        :param num_species: Number of chemical species in the network
        :n_desc: Number of descendants of K
        :conc_hidden/out: concentration of the species in the hidden/out layers
        """

        self.num_species = num_species
        self.num_input = num_input      # number of input nodes
        self.num_output = num_output    # number of output nodes 

        if (num_input + num_output > num_species):  
             raise ValueError("Input and output layers overlap")

        self.num_desc = n_desc

        self.concentration = np.ones(num_species) # total concentrations
        self.concentration[:num_species-num_output] = conc_hidden # the input concentrations are overwritten when running inference dynamics 
        self.concentration[num_species-num_output:] = conc_out
        self.conc_out = conc_out
        self.conc_hidden = conc_hidden
        self.conc_hidden_min = 0.2
        self.conc_hidden_max = 5
        self.list_conc_out = conc_out*np.ones(n_desc)
        self.list_conc_hidden = conc_hidden*np.ones(n_desc)

        # initialisation of the dynamic variables (affinities)
        self.states = [np.zeros(num_species) for _ in range(n_desc)] # neuronal variables for all descendants

        # interaction matrix 
        self.Ks = [np.zeros((num_species, num_species)) for _ in range(n_desc)]
        self.Ksymm = np.ones((num_species, num_species)) # used to enforce symmetry of Ks 
        for i in range(1,num_species):
            for j in range(0,i):
                self.Ksymm[i,j]=0.0

        # print('Ksymm=',self.Ksymm)
        self.Kmask = np.ones((num_species, num_species))
        for i in range(0,num_species): # used to set self-interactions = 0
            self.Kmask[i,i] = 0.0
        self.Kmask[:num_input,:num_input] = 0.0 # used to switch off the interaction between input species 
        # print('Kmask=',self.Kmask)

        # score ranks the different models 
        self.scores = np.zeros(n_desc) 

    def clone_K(self,mean,sigma): # this routine should be vectorized 
        K=self.Ks[0]
        for i in range(1,self.num_desc):
            normal_matrix = np.random.lognormal(mean=mean, sigma=sigma, size=(self.num_species,self.num_species))
            normal_matrix = normal_matrix*self.Kmask*self.Ksymm
            normal_matrix = normal_matrix+normal_matrix.T
            self.Ks[i] = np.minimum(K*normal_matrix,np.power(10.,4.))  

    def initialise_K(self,mean_0,sigma_0,mean,sigma):
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
            concs_hidden=conc_hidden*np.array([0.97,1.0,1.0/0.97])
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
        iopt=np.argmin(self.scores) # carefull, this argmax will pick up the first index when the scores are comparable
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
    def save(self, file_path):
        """
        Save the current ChemicalNetwork object to a file.
        
        :param file_path: Path to the file where the object should be saved.
        """
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, file_path):
        """
        Load a ChemicalNetwork object from a file.
        
        :param file_path: Path to the file from which the object should be loaded.
        :return: A ChemicalNetwork object.
        """
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj



#  --------------------------------
N_species=15
N_inp=6
N_out=3

N_desc=50+1
Conc_hidden=1.
Conc_out=1.
mean=-0 # mean of the normal distribution used when cloning (0.02)
sigma=0.5 # variance used when cloning
sigma_0=np.sqrt(2.) # variance of the starting patterns' distribution 
mean_0=-1.
# sigma_0=1 mu_0=-2.0
#D2=1
#D=np.sqrt(D2) # variance of the noise 


precision=np.power(10.,-6)
#  ---------------------------------

CN=ChemicalNetwork(N_species,N_inp,N_out,N_desc,Conc_hidden,Conc_out)


patterns_0=np.random.lognormal(mean=0, sigma=sigma_0,size=(3,6))              

if(unpert_pattern=='A'):
    patterns_0=np.array([[4.9 , 0.54 , 0.78 ,  0.11 ,  0.70 ,  1.5 ], 
     [3.9 ,  0.37 ,  0.33 ,  0.95 , 0.60 ,  1.4 ], 
     [3.1 ,  0.80 ,  0.07 ,  1.1 ,   0.18 , 26.8 ]])

if(unpert_pattern=='B'):
    patterns_0=np.array([[ 0.24001358,  3.19728681,  9.23301249,  1.44793741,  0.79277543,  0.56445251],
     [ 0.28378802,  1.98128614,  4.97609186,  0.19833512,  2.35384955,  0.09018742],
     [ 1.55839014,  0.52222897,  0.913423,    7.07690945, 41.08854987,  0.5581125 ]])
    
if(unpert_pattern=='C'):
    patterns_0=np.array([[ 1.32841165,  0.38709303,  2.29672817,  9.45836126, 20.61518202,  0.09458833],
 [33.60512186,  0.7034888,   2.28065926,  3.14203644,  5.32061224,  0.46844228],
 [ 0.58673294,  1.16767597,  8.43878063,  0.10142915,  0.69748406,  0.89055241]])
    
if(unpert_pattern=='D'):
    patterns_0=np.array([[0.57127289, 0.83179442, 2.14768857, 5.98053458, 0.69395479, 3.37960999],
 [2.27496627, 3.21611655, 7.33597589, 0.49539532, 3.31741288, 0.3586194 ],
 [3.83556362, 5.05860257, 2.5699809,  1.84195934, 0.19433368, 1.49518432]])

if(unpert_pattern=='E'):
    patterns_0=np.array([[0.99340207, 1.71808866, 2.1284392,  0.72005511, 1.12349243, 0.21264317],
 [0.04223733, 0.52639695, 3.86595848, 0.75410783, 0.50772593, 0.52794558],
 [1.53355982, 2.81928573, 0.61944862, 0.11851883, 3.14927676, 1.41633661]])

filename = f"patterns_{unpert_pattern}.pkl"
# with open(filename, 'wb') as f:
#     pickle.dump(patterns_0, f)
with open(filename, 'rb') as f:
    patterns_0=pickle.load(f)
print('unperturbed patterns=', patterns_0)

pp=0.0 # pp=0 for digital labels
labels_0 = np.array([[Conc_out*(10-pp)/10.,Conc_out*pp/10.,Conc_out*pp/10.],[Conc_out*pp/10.,Conc_out*(10.-pp)/10.,Conc_out*pp/10.],[Conc_out*pp/10.,Conc_out*pp/10.,Conc_out*(10.-pp)/10.]])     
dim_pattern=np.shape(patterns_0)[1]

# M_train/test makes no sense as the noise is resampled for each pattern used either in training or testing 
# N_patt_train=250
# N_patt_test=50

#def gen_patt_label(N_patt,D):
#    i=0
#    patterns=np.empty((0, 6))
#    labels=np.empty((0, 3))
#    while(i<N_patt):
#        nl=np.random.randint(0, np.shape(patterns_0)[0])
#        label=np.copy(labels_0[nl,:])
#        pattern=np.copy(patterns_0[nl,:])
#        pattern=pattern*np.random.lognormal(mean=0, sigma=D,size=dim_pattern)
#        patterns = np.vstack(([patterns, pattern]))
#        labels = np.vstack(([labels, label]))
#        i=i+1
#    return patterns, labels

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

# preparing the train dataset
#
# patterns_train, labels_train = gen_patt_label(N_patt_train,D)
#

# print(patterns_train)
# print(labels_train)

"""
:loss=0: cross entropy loss 'log[a1(c_out-a2)]'
:loss=1: unconventional loss 'log[a1/a2]'
:loss=2: MS loss 'a1^2+(c_out-a2)^2'
"""   
for drift in drift_scan:
    mean=-0.01*drift
    for noise in noise_scan:

        print(drift,noise)
        N2S=0.01*noise
        D=N2S*sigma_0
        filename = f"CN_{unpert_pattern}_{drift}_{noise}.pkl"
    
        if not os.path.exists(filename):
    # File does not exist: execute your code.
    
            n_batch=48 # use multiple of 3
            n_tot_batch=400

            loss=3
            CN.initialise_K(mean_0,sigma_0,mean,sigma)

            list_Ks=[]
            list_table_losses=[]

            # training
            i_batch=0

            list_a_patt1 = np.empty((0,np.shape(labels_0)[1]))
            list_a_patt2 = np.empty((0,np.shape(labels_0)[1]))
            list_a_patt3 = np.empty((0,np.shape(labels_0)[1]))
            list_a_var_patt1 = np.empty((0,np.shape(labels_0)[1]))
            list_a_var_patt2 = np.empty((0,np.shape(labels_0)[1]))
            list_a_var_patt3 = np.empty((0,np.shape(labels_0)[1]))
            list_score = np.array([])
            list_batch_pass = np.array([])

            list_conc_hidden = []

            # scatter plot for the untrained model 

            patterns_scatter, labels_scatter = gen_patt_label(5*n_batch,D)

            a1_0, a2_0, a3_0 = CN.test_model(patterns_scatter, labels_scatter, CN.conc_hidden, \
                                            CN.conc_out, precision)

            
            # np.savez(filename, arr1=a1_0, arr2=a2_0,arr3=a3_0)

            # training 

            while(i_batch<n_tot_batch):

                patterns, labels = gen_patt_label(n_batch,D)
                Conc_hidden = CN.conc_hidden
                Conc_out = CN.conc_out

                if(np.mod(10*i_batch,n_tot_batch)==0):
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
                if(np.mod(i_batch,1) == 0):
                    patterns_test, labels_test = gen_patt_label(4*n_batch,D) 
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
            

            #    print('list_a_patt2',list_a_patt2)
            #    print('---------------')

                if(np.mod(10*i_batch,n_tot_batch)==0):
                    list_Ks.append(CN.Ks[0])
                    print('i_batch=',i_batch)
            
            CN.save(filename)





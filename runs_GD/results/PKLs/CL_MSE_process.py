
import numpy as np
import pickle
import matplotlib.pyplot as plt
import h5py
import sys

index_run = int(sys.argv[1])
#index_run=1


noise_scan=range(10,121,5)
drift_scan = [0,2,4,8]  # you can add more drift values if needed

cocktail_scan = [index_run]
noise_scan = range(10, 121, 5)

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



def compute_mutual_I(x, i, dx=0.01):
    # Combine all channels together
    x_combined = np.concatenate([x[0], x[1], x[2]])
    x_i = x[i]
    
    # Define histogram edges
    edges = np.arange(np.min(x_combined), np.max(x_combined) + dx, dx)
    
    # Compute histograms for the selected index and for the combined data
    counts_i, _ = np.histogram(x_i, bins=edges, density=False)
    counts_combined, _ = np.histogram(x_combined, bins=edges, density=False)
    
    # Total counts for normalization
    total = np.sum(counts_combined)
    if total == 0:
        return 0
    
    # Only consider bins where counts_combined > 0
    valid = counts_combined > 0
    # Compute p_i for valid bins
    p_i = counts_i[valid] / counts_combined[valid]
    # Relative frequency for the combined data in the valid bins
    p = counts_combined[valid] / total
    
    # Only include indices where the ratio is strictly between 0 and 1
    valid_indices = (p_i > 0) & (p_i < 1)
    p_i = p_i[valid_indices]
    p = p[valid_indices]
    
    # Compute the mutual information using the given formula
    numerator = np.sum(p * (p_i * np.log(p_i) + (1 - p_i) * np.log(1 - p_i)))
    normalization = np.log(3) - (2/3) * np.log(2)
  
    
    return 1+numerator/normalization





def generate_svg_plots(a_list, colors, lower_bound=-6, ID=""):
    """
    Generates two distinct SVG plots:
      1. A 3D scatter plot (with shadows projected onto the xy-plane).
      2. Histograms of the f_x, f_y, and f_z components (stacked vertically).
    
    
    
    Both SVG outputs preserve editable text.
    
    Parameters:
      a_list       : list of numpy arrays, each with shape (n, 3) containing positive values.
      colors       : list of three color strings, one for each dataset.
      lower_bound  : fixed z-value for the xy-plane shadow and lower limit of the axes.
      scatter_filename : filename for the scatter plot SVG.
      hist_filename    : filename for the histograms SVG.
    """
    scatter_filename = f"SVGs/scatter_{ID}.svg"
    scatter_png = f"images/scatter_{ID}.png"
    hist_filename = f"SVGs/histograms_{ID}.svg"
    hist_png = f"images/histograms_{ID}.png"
   
    # Scaling factor: convert natural logs to base-10 logs.
    Dec = 1 / np.log(10)
    
    # Compute transformed data for each dataset.
    # Each a is assumed to have three columns: [f_x, f_y, f_z]
    all_x, all_y, all_z = [], [], []
    for a in a_list:
        x = Dec * np.log(a[:, 0])
        y = Dec * np.log(a[:, 1])
        z = Dec * np.log(a[:, 2])
        all_x.append(x)
        all_y.append(y)
        all_z.append(z)
  
    # Compute entropies for the x distributions.
    # H0: for first dataset; H1: for second; H2: for third.
 

    # Define tick positions from 0 to lower_bound in steps of -2.
    ticks = np.arange(0, lower_bound - 0.1, -1)  # subtract small value to ensure inclusion
    # Create tick labels: 0 is displayed as "1", other ticks as 10^{n}.
    labels = ["1" if tick == 0 else r'$10^{' + str(int(tick)) + '}$' for tick in ticks]
    font_size=28
    label_font=15

    # ======================================================
    # 1. Create the 3D Scatter Plot with Shadows and Export SVG
    # ======================================================
    fig_scatter = plt.figure(figsize=(10, 10))
    ax_scatter = fig_scatter.add_subplot(111, projection='3d')
   
    point_size = 10      # marker size for scatter points
    shadow_alpha = 0.05  # transparency for shadow points
    shadow_size = 1      # marker size for shadows
    
    # Plot each dataset in its color.
    for i, color in enumerate(colors):
        x = all_x[i]
        y = all_y[i]
        z = all_z[i]
        ax_scatter.scatter(x, y, z, color=color, s=point_size)
        # Plot shadow by re-plotting with z fixed at lower_bound.
        ax_scatter.scatter(x, y, zs=lower_bound, zdir='z', color=color, alpha=shadow_alpha, s=shadow_size)
    
    # Remove default axis labels; add custom axis names via text.
    ax_scatter.set_xlabel('')
    ax_scatter.set_ylabel('')
    ax_scatter.set_zlabel('')
    ax_scatter.text2D(0.27, 0.87, r'$f_x$', fontsize=font_size, transform=ax_scatter.transAxes)
    ax_scatter.text2D(0.73, 0.87, r'$f_y$', fontsize=font_size, transform=ax_scatter.transAxes)
    ax_scatter.text2D(1.0, 0.5, r'$f_z$', fontsize=font_size, transform=ax_scatter.transAxes)
    
    # Set limits for each axis.
    ax_scatter.set_xlim(lower_bound, 0)
    ax_scatter.set_ylim(lower_bound, 0)
    ax_scatter.set_zlim(lower_bound, 0)
    
    # Apply same ticks/labels on all three axes.
    ax_scatter.set_xticks(ticks)
    ax_scatter.set_xticklabels(labels, fontsize=label_font)
    ax_scatter.set_yticks(ticks)
    ax_scatter.set_yticklabels(labels,fontsize=label_font)
    ax_scatter.set_zticks(ticks)
    ax_scatter.set_zticklabels(labels,fontsize=label_font)
    
    ax_scatter.view_init(elev=20, azim=45)
    #ax_scatter.set_title("3D Scatter Plot with Shadows", fontsize=16)
    #fig_scatter.tight_layout()
    # Export scatter plot to SVG.
    print(scatter_filename)
    fig_scatter.savefig(scatter_filename, facecolor='none', transparent=True, format="svg")
    fig_scatter.savefig(scatter_png, facecolor='none', transparent=True, format="png")
    plt.close(fig_scatter)
    
    # ======================================================
    # 2. Create the Histograms Plot and Export SVG
    # ======================================================
    # Create a figure with three vertically stacked subplots.
    fig_hist = plt.figure(figsize=(8, 16))
    ax_hist_x = fig_hist.add_subplot(311)
    ax_hist_y = fig_hist.add_subplot(312)
    ax_hist_z = fig_hist.add_subplot(313)

    bins = 30  # number of bins for the histograms.
    
    # Plot histogram for f_x.
    for i, color in enumerate(colors):
        ax_hist_x.hist(all_x[i], bins=bins, color=color,density=True, alpha=0.7)
    # Plot histogram for f_y.
    for i, color in enumerate(colors):
        ax_hist_y.hist(all_y[i], bins=bins, color=color,density=True, alpha=0.7)
 
    for i, color in enumerate(colors):
        ax_hist_z.hist(all_z[i], bins=bins, color=color,density=True, alpha=0.7)
    
 
    for ax_hist in [ax_hist_x, ax_hist_y, ax_hist_z]:
        ax_hist.set_xlim(lower_bound, 0)
        ax_hist.set_xticks(ticks)
        ax_hist.set_xticklabels(labels, fontsize=label_font)
  
    
    ax_hist_x.text(0.05, 0.9, r'$f_x$', fontsize=font_size, transform=ax_hist_x.transAxes)
    ax_hist_y.text(0.05, 0.9, r'$f_y$', fontsize=font_size, transform=ax_hist_y.transAxes)
    ax_hist_z.text(0.05, 0.9, r'$f_z$', fontsize=font_size, transform=ax_hist_z.transAxes)

    mutual_information_x= compute_mutual_I(all_x,0)
    mutual_information_y= compute_mutual_I(all_y,1)
    mutual_information_z= compute_mutual_I(all_z,2)


    ax_hist_x.text(0.6, 0.9, f'$I_X = {mutual_information_x:.2f}$', fontsize=font_size, color='red', transform=ax_hist_x.transAxes)
    ax_hist_y.text(0.6, 0.9, f'$I_Y = {mutual_information_y:.2f}$', fontsize=font_size, color='blue', transform=ax_hist_y.transAxes)
    ax_hist_z.text(0.6, 0.9, f'$I_Z = {mutual_information_z:.2f}$', fontsize=font_size, color='green', transform=ax_hist_z.transAxes)

    
    fig_hist.tight_layout()
  
    fig_hist.savefig(hist_filename, format="svg")
    fig_hist.savefig(hist_png, format="png")
    plt.close(fig_hist)
    return mutual_information_x, mutual_information_y, mutual_information_z
    # Optionally, display the SVG files in a Jupyter Notebook.
    # disp.display(disp.SVG(filename=scatter_filename))
    # disp.display(disp.SVG(filename=hist_filename))
    

# Make sure ChemicalNetwork, gen_patt_label, generate_svg_plots are imported or defined appropriately

N_species = 15
N_inp = 6
N_out = 3

N_desc = 50 + 1
Conc_hidden = 1.
Conc_out = 1.
mean = -0.0  # mean of the normal distribution used when cloning (0.02)
sigma = 0.5  # variance used when cloning
sigma_0 = np.sqrt(2.)  # variance of the starting patterns' distribution 
mean_0 = -1.

pp = 0.0  # pp=0 for digital labels
labels_0 = np.array([
    [Conc_out * (10 - pp) / 10., Conc_out * pp / 10., Conc_out * pp / 10.],
    [Conc_out * pp / 10., Conc_out * (10 - pp) / 10., Conc_out * pp / 10.],
    [Conc_out * pp / 10., Conc_out * pp / 10., Conc_out * (10 - pp) / 10.]
])

precision = np.power(10., -6)

# Simulation parameters

colors = ['red', 'blue', 'green']


for drift in drift_scan:
    for cocktail_set in cocktail_scan:
           
        pattern_filename = f"patterns_F{cocktail_set}.pkl"
        with open(pattern_filename, 'rb') as f:
            patterns_0 = pickle.load(f)
            
        ns_list = []
        mi_x_list = []
        mi_y_list = []
        mi_z_list = []
            
        for noise in noise_scan:
            NS_ratio = noise * 0.01
            D = sigma_0 * NS_ratio
            ns_list.append(NS_ratio)
                
            ID = f"{cocktail_set}_{drift}_{noise}"
            CN_filename = f"CN_F{ID}.pkl"
            CN = ChemicalNetwork.load(CN_filename)
            print(f"Processing {CN_filename}")
                
            # Generate test patterns and labels
            patterns_scatter, labels_scatter = gen_patt_label(4200, D)
            a1, a2, a3 = CN.test_model(patterns_scatter, labels_scatter,
                                             CN.conc_hidden, CN.conc_out, precision)
                
                # Generate plots and/or compute mutual information values
            mi_x, mi_y, mi_z = generate_svg_plots([a1, a2, a3], colors, ID=ID)
                # Alternatively:



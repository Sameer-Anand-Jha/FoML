from matplotlib import pyplot as plt
import numpy as np
import random
random.seed(45)

num_coins = 100
def toss(num_trials):
    '''
    num_trials: number of trials to be performed.
    
    return a numpy array of size num_trials with each entry representing the number of heads found in each trial

    Use for loops to generate the numpy array and 'random.choice()' to simulate a coin toss
    
    NOTE: Do not use predefined functions to directly get the numpy array. 
    '''
    global num_coins
    results = []
    
    ## Write your code here
    for i in range(num_trials):
        heads = 0
        for j in range(num_coins):
            if random.choice([0, 1]) == 0:
                heads+=1
        results.append(heads)
    
    results = np.array(results)
    return results
    

def plot_hist(trial):
    '''
    trial: vector of values for a particular trial.

    plot the histogram for each trial.
    Use 'axs' from plt.subplots() function to create histograms. You can search about how to use it to plot histograms.

    Save the images in a folder named "histograms" in the current working directory.  
    '''
    fig, axs = plt.subplots(figsize =(10, 7), tight_layout=True)
    
    ## Write your code here
    axs.hist(trial, bins=100)
    plot_hist.counter += 1
    
    axs.set_xlabel("Coins")
    axs.set_ylabel("Number of heads")
    #axs.set_xticks(np.arange(0, 101, 10))
    axs.legend([f"Number of trials = {len(trial)}"])
    fig.savefig(f"./histograms/histogram_{plot_hist.counter}.png")
    # plt.show()
    plt.close(fig)
    
 
plot_hist.counter = 0
    

if __name__ == "__main__":
    num_trials_list = [10,100,1000,10000,100000]
    for num_trials in num_trials_list:
        heads_array = toss(num_trials)
        plot_hist(heads_array)

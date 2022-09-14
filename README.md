# ALACPD
## [Memory-free Online Change-point Detection: A Novel Neural Network Approach](https://arxiv.org/abs/2207.03932)

ALACPD exploits an LSTM-autoencoder-based neural network to perform unsupervised online CPD; it continuously adapts to the incoming samples without keeping the previously received input, thus being memory-free. 

### Usage
To run the code with the desired specifications you can use "run.sbatch" file.

### Datasets
To evaluate our model, we have used datasets offered by [Turing Change Point Dataset](https://github.com/alan-turing-institute/TCPD).

### CPD results on the run_log Dataset
Red lines depict the detected change-points:
![CPD](https://user-images.githubusercontent.com/18033908/190223301-040d7339-67a9-42ad-986b-17eccb49abd0.JPG)

### Reuirements
This code has been tested on Python 3.6 using the following libraries:

tensorflow                1.14.0  
numpy                     1.19.2  
scikit-learn              0.24.2

### Acknowledgements
The start of this implementation are [LSTNet](https://github.com/fbadine/LSTNet) and [OED](https://github.com/tungk/OED).



### Contact
email: z.atashgahi@utwente.nl

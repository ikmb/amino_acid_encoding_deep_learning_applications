# Amino Acid Encoding Deep Learning Applications
The repository contains the scripts and the models used for the paper amino acid encoding using deep learning application.
# Environments and dependencies
we recommend creating a new conda environment using the config file defined at: resource/config.yml
# Examples
All the models developed and mentioned in the paper can be retrained using the training script defined at CustomTrainingScripts directory.
### train a DPPI model 
```
$python trainDPPIModel.py -n 21 -d 8 -t 1 -g 0 -f 0.75 -o results/example_one
```           
the above commandline would create a DPPI model using the blueprint defined at Models/DPPIBluePrint.py with a learned embedding of size 8, train it 75% of the training data on the first GPU on the system and write the results to results/example_one. To mark the embedding frozen the following command can be executed
```
$python trainDPPIModel.py -n 21 -d 8 -t 0 -g 0 -f 0.75 -o results/example_one
``` 

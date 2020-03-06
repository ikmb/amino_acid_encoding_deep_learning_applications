# Amino Acid Encoding Deep Learning Applications
The repository contains the scripts and the models used for the paper amino acid encoding using deep learning application.
# Environments and dependencies
we recommend creating a new conda environment using the config file defined at: resource/config.yml
# Examples
All the models developed and mentioned in the paper can be retrained using the training script defined at CustomTrainingScripts directory.
### train a DPPI model with learned embedding
```
$python trainDPPIModel.py -n 21 -d 8 -t 1 -g 0 -f 0.75 -o results/example_one
```           

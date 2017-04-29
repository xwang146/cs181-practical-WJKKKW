Running Instruction
====================

This document will explain how to run this project in the following steps:

1. Environment setting: please run all the commands under python 3 environment

2. Default command:
```
python run.py
```
This will return all the scores in 400 epoches under the setting of SARSA with no modified epsilon and no gravity inference. The scores will store in the output .npy file as well as printing out in the terminal with average score at each epoch. 

2. More help on command line: 
run this command for more information on how to start the script:
```
python run.py -h
```

3. Run actual algorithm with parameters:
```
python run.py -m <NAME_OF_METHOD> -e <EPSILON_MODIFIED> -g <GRAVITY_INFERENCE>
```

4. Detailed explanation on parameters we need to input for evaluating the reinforcement learning:
    - NAME_OF_METHOD: pass in whether "SARSA" or "Q_Learning" (in exact words) to indicate which method we will use for this round.
    - EPSILON_MODIFIED: pass in whether "True" or "False" to indicate whether we are going to graduately decrease the influnce of epsilon in the policy.
    - GRAVITY_INFERECE: pass in whether "True" or "False" to indicate whether we are going to take gravity into account when learning the game. 

5. Example input:
```
python run.py -m Q_Learning -e True -g True
```
This will return all the scores in 400 epoches under the setting of modified epsilon Q_Learning with gravity inference. The scores will store in the output .npy file as well as printing out in the terminal with average score at each epoch. 

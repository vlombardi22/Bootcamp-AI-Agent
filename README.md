# Bootcamp-AI-Agent
A method for training skillnets
Installed on Ubuntu 20.04.5

requirments 
  torch=1.11.0
  keras-rl2
  numpy=1.23.2
  vizdoom=1.1.13

For vizdoom you will also need
apt install cmake git libboost-all-dev libsdl2-dev libopenal-dev 

tasks123_boot_a3c tasks 1-3
arguments
  control: Y/N 
  a2c: Y/N

task123_boot_a3c_base tasks 1-3
arguments:
  a2c: Y/N

task123_boot_ppo tasks 1-3
arguments
  control: Y/N
  
task123_ppo_base tasks 1-3
arguments: none 

task123_dqn_base tasks 1-3
arguments: none

task45_boot_a3c tasks 4-5
arguments
  control: Y/N 
  a2c: Y/N 
  task: 4 or 5
  
task45_boot_a3c_base tasks 1-3
arguments:
  a2c: Y/N
  task: 4 or 5

task45_boot_ppo tasks 1-3
arguments
  control: Y/N
  task: 4 or 5

task45_dqn tasks 1-3
arguments
  task: 4 or 5
  
task6_boot_a3c tasks 6
arguments
  control: Y/N 
  a2c: Y/N 
  
task6_boot_ppo tasks 6
arguments
  control: Y/N

task6_boot_a3c_base tasks 1-3
arguments:
  a2c: Y/N

task6_ppo_base tasks 6
arguments: none 

task6_dqn tasks 6
arguments: none

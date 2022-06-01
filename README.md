# stochastic-human-driving-policies-drl
This repository is the implementation of the paper "Deep Reinforcement Learning for Human-Like Driving Policies in Collision Avoidance Tasks of Self-Driving Cars" (now being reviewed after an extensive revision).
Note that this project also provides the GAIL algorithm implementation as a benchmark. The results will be published soon under the revised version. It also contains a unique simulator environment and data (which will be discussed later) for training so you can replicate the results.
In short, this project is designed to find human-like driving policies using the imitative learning approach we have developed. To this end, we demonstrate our approach to relatively simple tasks - static/dynamic obstacle avoidance tasks. In the article, you will find further discussions and limitations.

## Getting started
Tested on python 3.7.9 version. At the moment, you can only run this project on a Windows OS. 

1. Clone the repository and install the requirements
'''
pip install -r requirements.txt
'''

2. Extract data.zip. you will find there two files: 
a. under expert->gp (stands for Gaussian Process) you will find the GP_expert.csv to run the "PLAIN" and "MDN" algorithms mentioned in the paper.
b. under expert->demonstrations you will find the demonstrations.csv to run the "GAIL" algorithm that used as a benchmark comparison to our method.

3. Extract simulator.zip, the Unity simulator. You will find there three environments:
   a. training road (track 0)
   b. generalization road 1 (track 1)
   c. generalization road 2 (track 2)

To replicate the paper results you sholud train the agents on track 0 and test it on track 1 and 2.
To train the agent open the track 0 file and run on your terminal:
'''
python main.py --algorithm MDN --train
'''
Please be patient, this may take up to a week.

To test your agent open any track you like and run on your terminal:
'''
python main.py --algorithm MDN --test
'''

## Reference
If you find this work useful in your research, please cite:
'''
@article{emuna2020deep,
  title={Deep reinforcement learning for human-like driving policies in collision avoidance tasks of self-driving cars},
  author={Emuna, Ran and Borowsky, Avinoam and Biess, Armin},
  journal={arXiv preprint arXiv:2006.04218},
  year={2020}
}
'''

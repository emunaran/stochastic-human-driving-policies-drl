# stochastic-human-driving-policies-drl
This repository is the implementation of the paper "Deep Reinforcement Learning for Human-Like Driving Policies in Collision Avoidance Tasks of Self-Driving Cars" (now being reviewed after an extensive revision).
Note that this project also provides the GAIL algorithm implementation as a benchmark. The results will be published soon under the revised version. It also contains a unique simulator environment and data (which will be discussed later) for training so you can replicate the results.
In short, this project is designed to find human-like driving policies using imitation learning approach we have developed. To this end, we demonstrate our approach to relatively simple tasks - static/dynamic obstacle avoidance tasks. In the article, you will find further discussions and limitations.

## Getting started
Tested on python 3.7.9 version. At the moment, you can only run this project on a Windows OS. 

1. Clone the repository and install the requirements:
```
pip install -r requirements.txt
```

2. Extract data.zip. You will find there two files: 
   1. Under expert->gp (stands for Gaussian Process) you will find the GP_expert.csv to run the "PLAIN" and "MDN" algorithms mentioned in the paper.
   2. Under expert->demonstrations you will find the demonstrations.csv to run the "GAIL" algorithm that used as a benchmark comparison to our method.

3. Download simulator environment run files from the following link:
[simulator_run_files.zip](https://drive.google.com/file/d/1NNKRYMmtKLYGRXHg_d_8r6tHXsn53HF4/view?usp=sharing)
5. Extract simulator_run_files.zip to any preferred folder. You will find there two types of road (described in the paper):
   1. training road (track 0)
   2. generalization road (track 1)

While the training road contains one distribution of obstacles, the generalization road contains three - Random, Gauss, and Batch.

To replicate the paper results you sholud train the agents on track 0 and test it on track 1.
To train the agent open the track 0 file and run on your terminal:
```
python main.py --algorithm MDN --train
```
Please be patient, this may take up to a week.

To test your agent open any track you want and run on your terminal:
```
python main.py --algorithm MDN
```

## Reference
If you find this work useful in your research, please cite:
```
@article{emuna2020deep,
  title={Deep reinforcement learning for human-like driving policies in collision avoidance tasks of self-driving cars},
  author={Emuna, Ran and Borowsky, Avinoam and Biess, Armin},
  journal={arXiv preprint arXiv:2006.04218},
  year={2020}
}
```

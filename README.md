# stochastic-human-driving-policies-drl
This repository presents the implementation of the paper ["Deep Reinforcement Learning for Human-Like Driving Policies in Collision Avoidance Tasks of Self-Driving Cars"](https://arxiv.org/abs/2006.04218) and an extended paper titled ["Example-guided learning of stochastic human driving policies using deep reinforcement learning"](https://link.springer.com/article/10.1007/s00521-022-07947-2).
Additionally, it includes an implementation of the [Generative Adversarial Imitation Learning (GAIL)](https://arxiv.org/abs/1606.03476) (Ho, J., & Ermon, S., 2016) as a benchmark. The repository features a unique simulator environment (developed in Unity) and associated data for training, allowing the replication of the results.
In essence, this project focuses on the development of human-like driving policies through an imitation learning approach we have developed. To this end, we demonstrate our methodology to relatively simple tasks such as static and dynamic obstacle avoidance tasks. In the article, you will find further discussions and limitations.

<p align="center">
   <img width="550" alt="driving_simulation" src="https://github.com/emunaran/stochastic-human-driving-policies-drl/assets/22548214/1d6ba27b-fdfd-421d-a37e-934119adce21">
</p>

<p align="center">
   <img width="550" alt="driving_simulation_2" src="https://github.com/emunaran/stochastic-human-driving-policies-drl/assets/22548214/7cfd8c7b-2485-4e56-b145-74f9908c403d">
</p>

<p align="center">
   <img width="550" alt="driving_simulation_3" src="https://github.com/emunaran/stochastic-human-driving-policies-drl/assets/22548214/0a607110-ff62-449a-acb6-260a16a76596">
</p>

<p align="center">
   <img width="775" alt="GPs" src="https://github.com/emunaran/stochastic-human-driving-policies-drl/assets/22548214/d2920ad4-7dc7-4ab7-9f85-70b6abe80637">
</p>

## Getting started
Tested on Python 3.7.9 version. Currently, you can only run this project on a Windows OS. 

1. Clone the repository and install the requirements:
```
pip install -r requirements.txt
```

2. Extract data.zip. You will find there two files: 
   1. Under expert->gp (stands for Gaussian Process) you will find the GP_expert.csv to run the "PLAIN" and "MDN" algorithms mentioned in the paper.
   2. Under expert->demonstrations you will find the demonstrations.csv to run the "GAIL" algorithm that is used as a benchmark comparison to our method.

3. Download simulator environment run files from the following link:
[simulator_run_files.zip](https://drive.google.com/file/d/1NNKRYMmtKLYGRXHg_d_8r6tHXsn53HF4/view?usp=sharing)
5. Extract simulator_run_files.zip to any preferred folder. You will find there two types of roads (described in the paper):
   1. training road (track 0)
   2. generalization road (track 1)

While the training road contains one distribution of obstacles, the generalization road contains three - Random, Gauss, and Batch.

To replicate the paper results you should train the agents on track 0 and test it on track 1.
To train the agent open the track 0 file and run it on your terminal:
```
python main.py --algorithm MDN --train
```
Please be patient, this may take up to a week.

To test your agent open any track you want and run it on your terminal:
```
python main.py --algorithm MDN
```

## Reference
If you find this work useful in your research, please cite:
```
@article{emuna2022example,
  title={Example-guided learning of stochastic human driving policies using deep reinforcement learning},
  author={Emuna, Ran and Duffney, Rotem and Borowsky, Avinoam and Biess, Armin},
  journal={Neural Computing and Applications},
  pages={1--14},
  year={2022},
  publisher={Springer}
}
```

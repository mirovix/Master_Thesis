# Reinforcement learning in shared intelligence systems for mobile robots
In this thesis, we investigate how to integrate reinforcement learning in a shared
intelligence system where the user’s commands are equally fused with the
robot’s perception during the teleoperation of mobile robots. Specifically, we illustrate a new policy-based implementation suitable for navigating in unknown
indoor environment with the ability to avoid collisions with dynamic and static
obstacles. The aim of this thesis consists of extending the current system of
shared intelligence based on numerous pre-defined policies with a new one
based on Reinforcement Learning (RL).
To design this policy, an agent learns to reach a predefined goal by multiple
trial and error interactions. To make the robot learn correct actions, a reward
function is defined taking inspiration from the Attractive Potential Field (APF)
and the state is computed through a pre-processing module that clusters the
obstacles around the robot and finds the closest point with respect to the robot.
Different clustering algorithms are analysed for establishing which of them is
the most suitable for the purpose considering the real-time constraints required
by the system.
Different model configurations are examined and trained in gazebo-based simulation scenarios and then evaluated in different navigation scenarios. In this
way, we verified the reactive navigation behaviour of the agent with static and
dynamic obstacles. The shared system combined with the new RL policy is
tested and compared with the current state-of-the-art version, in a dedicated
teleoperated experiment where an operator was required to interact with the
robot by delivering high-level commands.

# Performance 
## Training
### Example of training

<p align="center">
<img height="400" src="https://github.com/mirovix/Master_Thesis/tree/main/readme_files/train_video.gif" title="Training scene" width="550"/>
</p>

### Performance of multiple agents

![](https://github.com/mirovix/Master_Thesis/tree/main/readme_files/train.png "Train plot")

(Top) Accumulated reward per episode achieved with the agents. (Bottom) The accumulated action-value function (Q) per episode achieved with the agents.

## Testing with policies
### Example of testing sequence

<p align="center">
<img height="350" src="https://github.com/mirovix/Master_Thesis/tree/main/readme_files/RL_test_video.gif" title="Testing scene" width="550"/>
</p>

### Path used for testing multiple systems

<p align="center">
<img height="550" src="https://github.com/mirovix/Master_Thesis/tree/main/readme_files/testing_path.png" title="testing path plot" width="350"/>
</p>

Trajectories of 𝑆𝐼 +𝑅𝐿𝑃4 (green path) and 𝑆𝐼 (red path) approaches.
The blue circles represent the target positions while the blue dotted lines are the
dynamic obstacles. The black elements are the obstacles or walls sensed by the
robot during the navigation.

### Test results of the examined shared intelligence systems  


<div align="center">

| System    | GA_P [%] | SA [%] | APL [m]   | MD [m]   | AT_P [m]   | ANUI | 
|-----------|----------|--------|-----------|----------|------------|------|
| SI+RLP1   | 76.19    | 85.71  | 43.1800   | 0.0658   | 1077.853   | 123  |
| SI+RLP2   | 80.95    | 85.71  | 37.3314   | 37.3314  | 0.0635     | 119  |
| SI+RLP3   | 85.71    | 95.23  | 39.3657   | 0.0615   | 830.5356   | 85   |
| `SI+RLP4` | `100`    | `100`  | 38.9604   | 38.9604  | `571.7233` | `63` |
| SI        | 90.48    | `100`  | `36.8820` | `0.0787` | 809.5110   | 84   |

</div>

#### Legend
- Goal Accuracy Policy (GA_P) represents the number of target positions reached over the total number of trials.
- Safety Accuracy (SA), this value, represents the probability of non-collision during the tests.
- Average Path Length (APL) measures the typical path length, estimated in meters.
- Mean Distance (MD) provides the measurement between the robot and the obstacles during the comparison test.
- Average Time Policy (AT_P) for completing the entire path (all 7 target positions).
- Average Number of User’s Input (ANUI) for completing the entire path.

# Usage
### Run training phase
```
roslaunch drone drone_train.launch path:="/home/miro/tiago_public_ws/src"
```
path >> directory where thesis_mihailovic is saved
### Run testing phase
```
roslaunch drone drone_test.launch path:="/home/miro/tiago_public_ws/src/drone/models/"
```
path >> directory of the main directory where the weights are saved
### Run policy fusion
```
roslaunch drone rl_policy.launch path:="/home/miro/tiago_public_ws/src/drone/models/"
```
path >> directory of the main directory where the weights are saved

## Library versions:

- python3
- tensorflow: 2.4.0
- pip: 21.3.1
- sklearn: 0.24.2
- numpy: 1.19.5
- scipy: 1.5.4
- matplotlib: 3.3.4

## Contact

miroljub mihailovic - miroljubmihailovic98@gmail.com - https://www.linkedin.com/in/miromiha
<div align="center">   
  
# ResCor: Residual Correct Safe Reinforcement Learning for Multi-agent Safety-critical Scenarios
</div>


# Abstract
Low-frequency but high-risk safety-critical events are the root of the long-tail effect in autonomous driving. When individual intelligence expands to multi-agent safety-critical scenarios (MASCS), their complexity surpasses single-agent theory and demands new collective-decision paradigms. Existing safe reinforcement learning (RL) methods, ranging from explicit safety masks to implicit Lagrangian constraints, improve ssfety yet either lock policies inside rigid safe sets or degrade efficiency through poor constraint representation. We introduce weak-to-strong generalisation into safe RL for the first time and propose **ResCor**, a **res**idual **cor**rection safe reinforcement learning framework that raises both safety and performance in MASCS. ResCor combines three modules: (i) a mixed-action policy that adaptively calibrates safety boundaries, (ii) a multi-agent dynamic conflict zone that captures and quantifies risk interactions, and (iii) a risk-aware prioritised experience replay that focuses learning on rare, high-risk events. Notably, the lightweight ResCor uses only 10% of the main model’s parameters yet provides decisive safety guidance without hindering task efficiency. Experiments on five MASCS show that ResCor cuts collision rates by up to **90.6\%** and increases cumulative rewards by 51.6% over strong baselines. It also transfers smoothly to unseen, more complex MASCS, highlighting its promise as a lightweight pre-trained safety module for real-world autonomous driving. 


# Methods
![method](Figs/frame.png "model frame")


# Demonstration Video
### 1. HardBrake (2AVs)


### 2. ParkingCrossingPed (2AVs)


### 3. ParkingCutIn


### 4. DynamicCutIn


### 5. IntersectionViolationPeds


### 6. HardBrake (3AVs)


### 7. ParkingCrossingPed (3AVs)



# Installation
```shell
# Clone the code to local
git clone https://github.com/Wangkaifeng-2021/ResCor.git
cd ResCor

# Create virtual environment
conda create -n ResCor python=3.7
conda activate ResCor

# Install basic dependency
pip install -r requirements.txt
```

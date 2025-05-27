import os
import traci
import torch
import numpy as np
from Experiment.PPO import create_ppo
from Experiment.PPO_ResCor import create_ppo_rescor
from Experiment.PPO_Lag import create_ppo_lag
import matplotlib.pyplot as plt
from main_HardBrake import start_sumo_env, get_valid_actions_mask, take_actions, graph_representation, \
    reward_function, compute_metrics, sim_params, safety_reward_function, risk_quantification_function, cost_function, \
    safety_layer

# ------ Experiment Param Config ------ #
Enable_Graph = True
# Enable_Graph = False

RENDER = True
# RENDER = False

# add noise during testing
sim_params['Noise_Level'] = 0
# sim_params['Noise_Level'] = 0.1
# sim_params['Noise_Level'] = 0.2

GNN = sim_params['GNN']

DRL = sim_params['DRL']

num_vehicles = sim_params['n_vehicles']
test_episodes = 20
random_seeds = list(range(test_episodes))

# path of the model tested
if Enable_Graph:
    load_dir = 'TrainedModels/HardBrake' + '/' + GNN + '_' + DRL + '/' + '2025_03_24-21_43_01'
else:
    load_dir = 'TrainedModels/HardBrake' + '/' + DRL + '/' + ''


if __name__ == "__main__":
    Rewards = []
    Col_Rewards = []
    Vel_Rewards = []
    Dis_Keep_Rewards = []
    Comfort_Rewards = []
    Avoid_Rewards = []
    LC_Rewards = []
    if DRL == 'PPO_ResCor':
        Safety_Rewards = []
        Safety_Col_Rewards = []
        Safety_Avoid_Rewards = []
    Episode_Steps = []
    Avg_Speeds = []
    Avg_Lat_Accs = []
    Avg_Lon_Accs = []
    Col_Nums = []

    if DRL == 'PPO' or DRL == 'PPO_SafetyLayer':
        GRL_Net, GRL_model = create_ppo(sim_params)
    elif DRL == 'PPO_ResCor':
        GRL_Net, GRL_model = create_ppo_rescor(sim_params)
    elif DRL == 'PPO_Lag':
        GRL_Net, GRL_model = create_ppo_lag(sim_params)
    else:
        raise ValueError(f"Unexpected value for DRL: {DRL}")
    GRL_model.load_model(load_dir)

    print("#-------------HardBrake--------------#")
    print("#-----------Testing Begins-----------#")
    print("#------------------------------------#")
    print(f"Enable_Graph: {Enable_Graph}, GNN: {GNN}, DRL: {DRL}")

    for i in range(1, test_episodes + 1):
        sim_params['AV00_is_avoided'] = False
        sim_params['AVs_lc'] = {'AV_0.0': {'step': -1, 'target_lane': -1},
                                'AV_1.0': {'step': -1, 'target_lane': -1},
                                'AV_2.0': {'step': -1, 'target_lane': -1}}
        random_seed = random_seeds[i-1]
        print(f"random_seed = {random_seed}")
        start_sumo_env(seed=random_seed, is_render=RENDER, params=sim_params)
        # Warm-up environment
        action = np.zeros(GRL_Net.num_agents, dtype=int)
        env_info = take_actions(action, sim_params, preheat=True, time_step=0)
        state = graph_representation(env_info, sim_params)

        Reward = np.zeros(num_vehicles)
        Col_Reward = np.zeros(num_vehicles)
        Vel_Reward = np.zeros(num_vehicles)
        Dis_Keep_Reward = np.zeros(num_vehicles)
        Comfort_Reward = np.zeros(num_vehicles)
        Avoid_Reward = np.zeros(num_vehicles)
        LC_Reward = np.zeros(num_vehicles)
        if DRL == 'PPO_ResCor':
            Safety_Reward = np.zeros(num_vehicles)
            Safety_Col_Reward = np.zeros(num_vehicles)
            Safety_Avoid_Reward = np.zeros(num_vehicles)
        Step_Avg_Speeds = []
        Step_Avg_Lat_Accs = []
        Step_Avg_Lon_Accs = []
        Collision_Num = 0
        steps = 0
        horizon = sim_params['horizon']
        for step in range(horizon):
            risk = risk_quantification_function(env_info, sim_params)
            actions_mask = get_valid_actions_mask(state, env_info, step, sim_params)
            # ------ choose actions ------ #
            if DRL == 'PPO_ResCor':
                action = GRL_model.test_action(state, actions_mask, risk)
            else:
                action = GRL_model.test_action(state, actions_mask)
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            if DRL == 'PPO_SafetyLayer':
                action = safety_layer(state, env_info, action, sim_params)
            env_info = take_actions(action, sim_params, preheat=False, time_step=step)
            steps += 1
            next_state = graph_representation(env_info, sim_params)
            # ------ reward function ------ #
            reward, col_reward, vel_reward, dis_keep_reward, com_reward, avd_reward, lc_reward \
                = reward_function(env_info, sim_params)
            if DRL == 'PPO_ResCor':
                safety_reward, safety_col_reward, safety_avd_reward = safety_reward_function(env_info, sim_params)
            if DRL == 'PPO_Lag':
                cost, col_cost, ttc_cost = cost_function(env_info, sim_params)
                avd_reward = col_cost  # for convenience, directly replace
                lc_reward = ttc_cost
            avg_speed, avg_lat_acc, avg_lon_acc = compute_metrics(env_info)
            col_num = env_info['collision_num']
            done = False
            remain_number = traci.simulation.getMinExpectedNumber()
            if remain_number == 0:
                print(f"simulation over, Episode: {i}, steps: {steps}")
                done = True
            else:
                veh_ids = traci.vehicle.getIDList()
                ids_set = set(veh_ids)
                if ids_set == {'HV_0.0'}:
                    done = True
            # episode reward
            Reward += reward
            Col_Reward += col_reward
            Vel_Reward += vel_reward
            Dis_Keep_Reward += dis_keep_reward
            Comfort_Reward += com_reward
            Avoid_Reward += avd_reward
            LC_Reward += lc_reward
            if DRL == 'PPO_ResCor':
                Safety_Reward += safety_reward
                Safety_Col_Reward += safety_col_reward
                Safety_Avoid_Reward += safety_avd_reward
            Collision_Num += col_num
            if env_info['has_av']:
                Step_Avg_Speeds.append(avg_speed)
                Step_Avg_Lat_Accs.append(avg_lat_acc)
                Step_Avg_Lon_Accs.append(avg_lon_acc)
            state = next_state
            if done:
                break
        traci.close()
        # ------ record testing data ------ #
        Episode_Steps.append(steps)
        # normalize reward
        Reward /= steps
        Col_Reward /= steps
        Vel_Reward /= steps
        Dis_Keep_Reward /= steps
        Comfort_Reward /= steps
        Avoid_Reward /= steps
        LC_Reward /= steps
        if DRL == 'PPO_ResCor':
            Safety_Reward /= steps
            Safety_Col_Reward /= steps
            Safety_Avoid_Reward /= steps
        num_av = sim_params['n_av']
        Reward_Sum_AVs = np.sum(Reward[:num_av])
        Col_Reward_Sum_AVs = np.sum(Col_Reward[:num_av])
        Vel_Reward_Sum_AVs = np.sum(Vel_Reward[:num_av])
        Dis_Keep_Reward_Sum_AVs = np.sum(Dis_Keep_Reward[:num_av])
        Comfort_Reward_Sum_AVs = np.sum(Comfort_Reward[:num_av])
        Avoid_Reward_Sum_AVs = np.sum(Avoid_Reward[:num_av])
        LC_Reward_Sum_AVs = np.sum(LC_Reward[:num_av])
        if DRL == 'PPO_ResCor':
            Safety_Reward_Sum_AVs = np.sum(Safety_Reward[:num_av])
            Safety_Col_Reward_Sum_AVs = np.sum(Safety_Col_Reward[:num_av])
            Safety_Avoid_Reward_Sum_AVs = np.sum(Safety_Avoid_Reward[:num_av])
        Rewards.append(Reward_Sum_AVs)
        Col_Rewards.append(Col_Reward_Sum_AVs)
        Vel_Rewards.append(Vel_Reward_Sum_AVs)
        Dis_Keep_Rewards.append(Dis_Keep_Reward_Sum_AVs)
        Comfort_Rewards.append(Comfort_Reward_Sum_AVs)
        Avoid_Rewards.append(Avoid_Reward_Sum_AVs)
        LC_Rewards.append(LC_Reward_Sum_AVs)
        if DRL == 'PPO_ResCor':
            Safety_Rewards.append(Safety_Reward_Sum_AVs)
            Safety_Col_Rewards.append(Safety_Col_Reward_Sum_AVs)
            Safety_Avoid_Rewards.append(Safety_Avoid_Reward_Sum_AVs)
        Episode_Avg_Speed = 0.0
        Episode_Avg_Lat_Acc = 0.0
        Episode_Avg_Lon_Acc = 0.0
        if len(Step_Avg_Speeds) > 0:
            Episode_Avg_Speed = sum(Step_Avg_Speeds) / len(Step_Avg_Speeds)
        if len(Step_Avg_Lat_Accs) > 0:
            Episode_Avg_Lat_Acc = sum(Step_Avg_Lat_Accs) / len(Step_Avg_Lat_Accs)
        if len(Step_Avg_Lon_Accs) > 0:
            Episode_Avg_Lon_Acc = sum(Step_Avg_Lon_Accs) / len(Step_Avg_Lon_Accs)
        Avg_Speeds.append(Episode_Avg_Speed)
        Avg_Lat_Accs.append(Episode_Avg_Lat_Acc)
        Avg_Lon_Accs.append(Episode_Avg_Lon_Acc)
        Col_Nums.append(Collision_Num)
        print(f'Episode: {i}, steps: {steps}, Reward: {Reward_Sum_AVs:.1f}, '
              f'Episode_Avg_Speed: {Episode_Avg_Speed:.1f}, Collision_Num: {Collision_Num}')
        RED = '\033[91m'
        GREEN = '\033[92m'
        RESET = '\033[0m'
        col_reward_color = RED if Col_Reward_Sum_AVs < 0 else GREEN
        print(f'Col_Reward: {col_reward_color}{Col_Reward_Sum_AVs:.1f}{RESET}, '
              f'Vel_Reward: {Vel_Reward_Sum_AVs:.1f}, '
              f'Dis_Keep_Reward: {Dis_Keep_Reward_Sum_AVs:.2f}, '
              f'Comfort_Reward: {Comfort_Reward_Sum_AVs:.1f}, '
              f'Avoid_Reward: {Avoid_Reward_Sum_AVs:.2f}, '
              f'LC_Reward: {LC_Reward_Sum_AVs:.1f}')
        if DRL == 'PPO_ResCor':
            print(f'Safety_Reward: {Safety_Reward_Sum_AVs:.1f}, '
                  f'Safety_Col_Reward: {Safety_Col_Reward_Sum_AVs:.1f}, '
                  f'Safety_Avoid_Reward: {Safety_Avoid_Reward_Sum_AVs:.2f}')
        print('=' * 115)
        # plot
        if DRL == 'PPO_ResCor':
            plt.figure(num=1, figsize=(16, 7))
            plt.subplot(2, 5, 1)
            plt.title('Rewards')
            plt.plot(Rewards)
            plt.subplot(2, 5, 2)
            plt.title('Episode_Steps')
            plt.plot(Episode_Steps)
            plt.subplot(2, 5, 3)
            plt.title('Col_Rewards')
            plt.plot(Col_Rewards)
            plt.subplot(2, 5, 4)
            plt.title('Vel_Rewards')
            plt.plot(Vel_Rewards)
            plt.subplot(2, 5, 5)
            plt.title('Dis_Keep_Rewards')
            plt.plot(Dis_Keep_Rewards)
            plt.subplot(2, 5, 6)
            plt.title('Comfort_Rewards')
            plt.plot(Comfort_Rewards)
            plt.subplot(2, 5, 7)
            plt.title('Avoid_Rewards')
            plt.plot(Avoid_Rewards)
            plt.subplot(2, 5, 8)
            plt.title('Safety_Rewards')
            plt.plot(Safety_Rewards)
            plt.subplot(2, 5, 9)
            plt.title('Safety_Col_Rewards')
            plt.plot(Safety_Col_Rewards)
            plt.subplot(2, 5, 10)
            plt.title('Safety_Avoid_Rewards')
            plt.plot(Safety_Avoid_Rewards)
        else:
            plt.figure(num=1, figsize=(13, 7))
            plt.subplot(2, 4, 1)
            plt.title('Rewards')
            plt.plot(Rewards)
            plt.subplot(2, 4, 2)
            plt.title('Episode_Steps')
            plt.plot(Episode_Steps)
            plt.subplot(2, 4, 3)
            plt.title('Col_Rewards')
            plt.plot(Col_Rewards)
            plt.subplot(2, 4, 4)
            plt.title('Vel_Rewards')
            plt.plot(Vel_Rewards)
            plt.subplot(2, 4, 5)
            plt.title('Dis_Keep_Rewards')
            plt.plot(Dis_Keep_Rewards)
            plt.subplot(2, 4, 6)
            plt.title('Comfort_Rewards')
            plt.plot(Comfort_Rewards)
            plt.subplot(2, 4, 7)
            plt.title('Avoid_Rewards (Col_Costs)')
            plt.plot(Avoid_Rewards)
            plt.subplot(2, 4, 8)
            plt.title('LC_Rewards (TTC_Costs)')
            plt.plot(LC_Rewards)
        plt.subplots_adjust(hspace=0.5, wspace=0.5)
        plt.show(block=False)
        plt.pause(1)
    print('Testing Finished.')

    success_num = Col_Nums.count(0)
    print(f'success times/total times = {success_num}/{test_episodes}')

    save_dir = load_dir + f"/Test_Noise={sim_params['Noise_Level']:.2f}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(save_dir + "/Episode_Steps", Episode_Steps)
    np.save(save_dir + "/Rewards", Rewards)
    np.save(save_dir + "/Col_Rewards", Col_Rewards)
    np.save(save_dir + "/Vel_Rewards", Vel_Rewards)
    np.save(save_dir + "/Dis_Keep_Rewards", Dis_Keep_Rewards)
    np.save(save_dir + "/Comfort_Rewards", Comfort_Rewards)
    np.save(save_dir + "/Avoid_Rewards", Avoid_Rewards)
    np.save(save_dir + "/LC_Rewards", LC_Rewards)
    if DRL == 'PPO_ResCor':
        np.save(save_dir + "/Safety_Rewards", Safety_Rewards)
        np.save(save_dir + "/Safety_Col_Rewards", Safety_Col_Rewards)
        np.save(save_dir + "/Safety_Avoid_Rewards", Safety_Avoid_Rewards)
    np.save(save_dir + "/Avg_Speeds", Avg_Speeds)
    np.save(save_dir + "/Col_Nums", Col_Nums)

    plt.figure(1)
    plt.savefig(save_dir + '/Test_Data_Figure.png')

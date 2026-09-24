import copy
import math

import numpy as np


class Environ:
    def __init__(self, sim_dict_init):
        self.bandwidth = int(5e4)
        self.sig2 = 1e-8
        self.task_on_base = sim_dict_init["task_on_base"]
        self.task_on_base_matrix = sim_dict_init["task_on_base_matrix"]
        self.fixed_distance_matrix = sim_dict_init["fixed_distance_matrix"]
        self.usable_channel_of_all_nodes = sim_dict_init["usable_channel_of_all_nodes"]
        self.task_deadline = np.asarray(sim_dict_init["task_deadline"], dtype=np.float64)
        self.task_data_size = np.asarray(sim_dict_init["task_data_size"], dtype=np.float64)
        self.power_choose = [0, 100, 200, 300, 400]
        self.n_agents = len(self.task_on_base)
        self.n_tasks = len(self.task_data_size)
        self.n_channels = max(len(ch) for ch in self.usable_channel_of_all_nodes) if self.usable_channel_of_all_nodes else 6
        all_channel_ids = [cid for ch_list in self.usable_channel_of_all_nodes for cid in ch_list]
        self.n_channel_ids = max(all_channel_ids) + 1 if all_channel_ids else 10
        self.V2V_Interference_all = np.zeros((self.n_agents, self.n_channels)) + self.sig2
        self.task_length = self.n_tasks
        self.PATH_LOSS_EXPONENT = 3.7
        self.t = 0
        self.overhead = 0
        self.total_data = 0
        self.total_time = 0

    def get_channel_choose(self, channel_index, task_index, channel_strategy):
        strategy_node = []
        while True:
            quotient = channel_strategy % (len(task_index) + 1)
            channel_strategy = channel_strategy // (len(task_index) + 1)
            strategy_node.append(quotient)
            if channel_strategy == 0:
                break
        if len(strategy_node) < len(channel_index):
            for i in range(len(channel_index) - len(strategy_node)):
                strategy_node.append(0)
        strategy_node.reverse()
        for i in range(len(strategy_node)):
            if strategy_node[i] == 0:
                strategy_node[i] = -1
            else:
                strategy_node[i] = task_index[int(strategy_node[i] - 1)]
        return strategy_node

    def get_power_choose(self, task_index, power_strategy):
        if len(task_index) == 0:
            return []
        if power_strategy < 0:
            power_strategy = 0
        strategy_mgt = []
        while True:
            quotient = int(power_strategy % len(self.power_choose))
            power_strategy = power_strategy // len(self.power_choose)
            strategy_mgt.append(quotient)
            if power_strategy == 0:
                break
        if len(strategy_mgt) < len(task_index):
            for i in range(len(task_index) - len(strategy_mgt)):
                strategy_mgt.append(0)
        strategy_mgt.reverse()
        strategy_mgt_test = np.zeros(len(task_index))
        for i in range(len(strategy_mgt)):
            strategy_mgt_test[i] = self.power_choose[strategy_mgt[i]]
        return strategy_mgt_test

    def Compute_Interference_2(self, distance_matrix, channel_index, strategy_channel_allocation, power_every_task):
        Infer_every_channel_tmp = np.zeros((self.n_agents, self.n_channel_ids))
        Infer_every_channel = [[] for _ in range(self.n_agents)]
        for i in range(len(channel_index)):
            for index, j in enumerate(channel_index[i]):
                for m in range(len(channel_index)):
                    if m != i:
                        for index_2, n in enumerate(channel_index[m]):
                            if n == j and strategy_channel_allocation[m][index_2] != -1:
                                Infer_every_channel_tmp[i][j] += (
                                    math.pow(8, 2)
                                    * power_every_task[strategy_channel_allocation[m][index_2]]
                                    * math.pow(
                                        distance_matrix[i][strategy_channel_allocation[m][index_2]],
                                        1 - self.PATH_LOSS_EXPONENT,
                                    )
                                )
        for i in range(len(channel_index)):
            for j in channel_index[i]:
                Infer_every_channel[i].append(Infer_every_channel_tmp[i][j])
        self.V2V_Interference_all = Infer_every_channel + self.sig2 * np.ones((self.n_agents, self.n_channels))

    def get_inference(self, distance_matrix, channel_index, strategy_channel_allocation, i, task_no, power_every_task):
        task_on_channel = []
        for j in range(len(strategy_channel_allocation[i])):
            if task_no == strategy_channel_allocation[i][j]:
                task_on_channel.append(channel_index[i][j])
        inf_sum = []
        for t in range(len(task_on_channel)):
            inf = 0
            for n in range(len(channel_index)):
                if n != i:
                    if task_on_channel[t] in channel_index[n]:
                        for j in range(len(channel_index[n])):
                            if channel_index[n][j] == task_on_channel[t]:
                                task_tmp = strategy_channel_allocation[n][j]
                                if (task_tmp != -1) & (task_tmp != -2):
                                    task_power = power_every_task[task_tmp]
                                    if distance_matrix[i][task_tmp] < 500:
                                        inf += (
                                            task_power
                                            * math.pow(8, 2)
                                            * math.pow(500, 0 - self.PATH_LOSS_EXPONENT)
                                        )
                                    else:
                                        inf += (
                                            task_power
                                            * math.pow(8, 2)
                                            * math.pow(
                                                distance_matrix[i][task_tmp],
                                                0 - self.PATH_LOSS_EXPONENT,
                                            )
                                        )
            inf_sum.append(inf)
        return inf_sum, task_on_channel

    def renew_task_on_base(self):
        task_tmp = [[], [], [], [], []]
        for i in range(len(self.task_on_base)):
            for j in self.task_on_base[i]:
                if j != -1:
                    task_tmp[i].append(j)
        self.task_on_base = task_tmp

    def get_current_reward(self, new_state, last_state):
        data_size_translation = 0
        for i in range(len(new_state)):
            data_size_translation += (last_state[i] - new_state[i]) / 5e5
        return data_size_translation

    def step(self, channel_strategy, power_strategy):
        strategy_channel_allocation = []
        strategy_power_mgt = []
        task_to_channel = []
        task_length = self.task_length
        task_to_base = np.zeros(task_length)
        last_data_size = self.task_data_size.copy()
        total_reward = 0
        success = 0
        fail = 0
        punish = 0
        power_all = 0
        data_size_all = 0
        task_on_base_tmp = copy.deepcopy(self.task_on_base)
        task_channel = []
        for i in range(task_length):
            task_channel_tmp = []
            task_channel.append(task_channel_tmp)
        action = []
        action.append(channel_strategy)
        action.append(power_strategy)
        if (not (np.any(self.task_deadline) == 0)) & (not (np.any(self.task_data_size) == 0)):
            for i in range(task_length):
                tmp = []
                task_to_channel.append(tmp)
            for i in range(len(task_on_base_tmp)):
                for j in task_on_base_tmp[i]:
                    task_to_base[j] = i
            for i in range(len(channel_strategy)):
                strategy_channel_allocation.append(
                    self.get_channel_choose(
                        self.usable_channel_of_all_nodes[i],
                        self.task_on_base[i],
                        channel_strategy[i],
                    )
                )
                strategy_power_mgt.append(
                    self.get_power_choose(self.task_on_base[i], power_strategy[i])
                )
            for i in range(len(self.usable_channel_of_all_nodes)):
                for j in range(len(self.usable_channel_of_all_nodes[i])):
                    if strategy_channel_allocation[i][j] != -1:
                        task_channel[strategy_channel_allocation[i][j]].append(
                            self.usable_channel_of_all_nodes[i][j]
                        )
            new_channel_task_matrix = np.zeros((task_length, self.n_channel_ids))
            for i in range(len(self.usable_channel_of_all_nodes)):
                for j in range(len(self.usable_channel_of_all_nodes[i])):
                    if strategy_channel_allocation[i][j] != -1:
                        new_channel_task_matrix[strategy_channel_allocation[i][j]][
                            self.usable_channel_of_all_nodes[i][j]
                        ] = 1
            power_every_task = np.zeros(self.task_length)
            for i in range(len(strategy_power_mgt)):
                for j in range(len(strategy_power_mgt[i])):
                    power_every_task[task_on_base_tmp[i][j]] = strategy_power_mgt[i][j]
            S = np.zeros(self.task_length)
            v = np.zeros(self.task_length)
            self.Compute_Interference_2(
                self.fixed_distance_matrix,
                self.usable_channel_of_all_nodes,
                strategy_channel_allocation,
                power_every_task,
            )
            for i in range(len(task_on_base_tmp)):
                for j in range(len(task_on_base_tmp[i])):
                    if task_on_base_tmp[i][j] not in strategy_channel_allocation[i]:
                        S[task_on_base_tmp[i][j]] = 0
                    else:
                        if self.fixed_distance_matrix[i][self.task_on_base[i][j]] < 500:
                            S[task_on_base_tmp[i][j]] = (
                                strategy_power_mgt[i][j]
                                * math.pow(8, 2)
                                * math.pow(
                                    self.fixed_distance_matrix[i][task_on_base_tmp[i][j]],
                                    0 - self.PATH_LOSS_EXPONENT,
                                )
                            )
                        else:
                            S[task_on_base_tmp[i][j]] = (
                                strategy_power_mgt[i][j]
                                * math.pow(8, 2)
                                * math.pow(500, 0 - self.PATH_LOSS_EXPONENT)
                            )
                        total_reward -= (
                            strategy_power_mgt[i][j]
                            * 200
                            * len(task_channel[task_on_base_tmp[i][j]])
                            / 5e5
                        )
                        power_all += (
                            strategy_power_mgt[i][j]
                            * 200
                            * len(task_channel[task_on_base_tmp[i][j]])
                            / 5e5
                        )
                        if S[task_on_base_tmp[i][j]] != 0:
                            self.overhead += 0.2
                            self.overhead += (
                                strategy_power_mgt[i][j]
                                * 200
                                * len(task_channel[task_on_base_tmp[i][j]])
                                / 1e5
                            )
                            inf, task_on_channel = self.get_inference(
                                self.fixed_distance_matrix,
                                self.usable_channel_of_all_nodes,
                                strategy_channel_allocation,
                                i,
                                self.task_on_base[i][j],
                                power_every_task,
                            )
                            for t in range(len(inf)):
                                task_to_channel[task_on_base_tmp[i][j]] = task_on_channel
                                SINR = S[task_on_base_tmp[i][j]] / (inf[t] + self.sig2)
                                if SINR > 2:
                                    self.total_time += 200
                                    tmp_task_data = 0
                                    v[task_on_base_tmp[i][j]] += round(
                                        self.bandwidth * np.log2(1 + SINR), 3
                                    )
                            tmp_task_data = self.task_data_size[task_on_base_tmp[i][j]]
                            self.task_data_size[task_on_base_tmp[i][j]] = round(
                                (
                                    self.task_data_size[task_on_base_tmp[i][j]]
                                    - v[task_on_base_tmp[i][j]] * 0.2
                                ),
                                3,
                            )
                            if (
                                self.task_data_size[task_on_base_tmp[i][j]] <= 0
                                & (
                                    self.task_deadline[task_on_base_tmp[i][j]] / 1000.0
                                    - tmp_task_data / v[task_on_base_tmp[i][j]]
                                    >= 0
                                )
                            ):
                                self.task_data_size[task_on_base_tmp[i][j]] = 0
                                self.task_deadline[task_on_base_tmp[i][j]] = 0
                                self.total_time -= (
                                    200 - (tmp_task_data / v[task_on_base_tmp[i][j]]) * 1000
                                )
                                tmp = (
                                    (0.2 - tmp_task_data / v[task_on_base_tmp[i][j]])
                                    * strategy_power_mgt[i][j]
                                    / 500
                                )
                                total_reward += (
                                    (0.2 - tmp_task_data / v[task_on_base_tmp[i][j]])
                                    * strategy_power_mgt[i][j]
                                    / 500
                                )
                                self.overhead -= (
                                    (2 - tmp_task_data / v[task_on_base_tmp[i][j]] * 10)
                                    + (
                                        (0.2 - tmp_task_data / v[task_on_base_tmp[i][j]])
                                        * strategy_power_mgt[i][j]
                                        / 100
                                    )
                                )
                                power_all -= tmp
                                self.task_deadline[task_on_base_tmp[i][j]] = 200
                                total_reward += 5
                                self.task_on_base[i][j] = -1
                                success += 1
                            elif (self.task_data_size[task_on_base_tmp[i][j]] <= 0) & (
                                self.task_deadline[task_on_base_tmp[i][j]] / 1000.0
                                - tmp_task_data / v[task_on_base_tmp[i][j]]
                                < 0
                            ):
                                self.task_data_size[task_on_base_tmp[i][j]] = 0
                                self.task_deadline[task_on_base_tmp[i][j]] = 200
                                fail += 1
                                self.task_on_base[i][j] = -1
                    self.task_deadline[task_on_base_tmp[i][j]] -= 200
                    if (self.task_deadline[task_on_base_tmp[i][j]] < 0) & (
                        self.task_data_size[task_on_base_tmp[i][j]] > 0
                    ):
                        self.task_deadline[task_on_base_tmp[i][j]] = 0
                        punish += self.task_data_size[task_on_base_tmp[i][j]]
                        self.task_data_size[task_on_base_tmp[i][j]] = 0
                        total_reward -= 5
                        fail += 1
                        self.task_on_base[i][j] = -1
            self.channel_task_matrix = new_channel_task_matrix
            task_on_base_tmp = self.task_on_base.copy()
            self.renew_task_on_base()
            self.total_data += (
                self.get_current_reward(self.task_data_size, last_data_size) * 5e5
            )
            total_reward += (
                self.get_current_reward(self.task_data_size, last_data_size)
                - punish / 5e5
            )
            data_size_all += (
                self.get_current_reward(self.task_data_size, last_data_size)
                - punish / 5e5
            )
        return total_reward, success, fail


class DiscreteEnviron(Environ):
    """Original environment with a direct discrete-action adapter.

    The original environment dynamics and reward are intentionally reused.
    This adapter converts decomposed discrete actions back to the original
    strategy IDs so experiments remain comparable with the paper baseline.
    """

    max_tasks_per_agent = 6
    max_channels_per_agent = 6

    @staticmethod
    def _encode_base_digits(digits, base):
        strategy = 0
        for digit in digits:
            strategy = strategy * base + int(digit)
        return int(strategy)

    def _encode_channel_strategy(self, agent_id, channel_choices):
        task_count = len(self.task_on_base[agent_id])
        channel_count = len(self.usable_channel_of_all_nodes[agent_id])
        base = task_count + 1
        digits = []
        for channel_idx in range(channel_count):
            choice = int(channel_choices[channel_idx])
            if choice < 0 or choice > task_count:
                choice = 0
            digits.append(choice)
        return self._encode_base_digits(digits, base)

    def _encode_power_strategy(self, agent_id, power_choices):
        task_count = len(self.task_on_base[agent_id])
        base = len(self.power_choose)
        digits = []
        for task_idx in range(task_count):
            choice = int(power_choices[task_idx])
            if choice < 0 or choice >= base:
                choice = 0
            digits.append(choice)
        return self._encode_base_digits(digits, base)

    def act_for_training_discrete(self, channel_choices, power_choices):
        """Run one environment step from decomposed discrete actions.

        Parameters
        ----------
        channel_choices:
            Array-like [n_agents, max_channels]. Each entry is a local task
            slot: 0 for idle, 1..len(task_on_base[i]) for a task.
        power_choices:
            Array-like [n_agents, max_tasks]. Each entry is an index into
            self.power_choose for the corresponding local task slot.
        """
        channel_choices = np.asarray(channel_choices)
        power_choices = np.asarray(power_choices)
        channel_strategy = []
        power_strategy = []
        for agent_id in range(len(self.task_on_base)):
            channel_strategy.append(
                self._encode_channel_strategy(agent_id, channel_choices[agent_id])
            )
            power_strategy.append(
                self._encode_power_strategy(agent_id, power_choices[agent_id])
            )
        return self.step(channel_strategy, power_strategy)

    def get_action_masks(self):
        """Return valid action masks for decomposed discrete actor heads."""
        n_agents = len(self.task_on_base)
        n_power = len(self.power_choose)
        channel_masks = np.zeros(
            (n_agents, self.max_channels_per_agent, self.max_tasks_per_agent + 1),
            dtype=np.float32,
        )
        power_masks = np.zeros(
            (n_agents, self.max_tasks_per_agent, n_power), dtype=np.float32
        )

        for agent_id in range(n_agents):
            task_count = min(
                len(self.task_on_base[agent_id]), self.max_tasks_per_agent
            )
            channel_count = min(
                len(self.usable_channel_of_all_nodes[agent_id]),
                self.max_channels_per_agent,
            )

            for channel_idx in range(channel_count):
                channel_masks[agent_id, channel_idx, : task_count + 1] = 1.0
            for channel_idx in range(channel_count, self.max_channels_per_agent):
                channel_masks[agent_id, channel_idx, 0] = 1.0

            for task_idx in range(task_count):
                power_masks[agent_id, task_idx, :] = 1.0
            for task_idx in range(task_count, self.max_tasks_per_agent):
                power_masks[agent_id, task_idx, 0] = 1.0

        return channel_masks, power_masks

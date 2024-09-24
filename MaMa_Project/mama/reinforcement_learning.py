import random
from typing import List, Dict

class AgentPolicy:
    """Defines a reinforcement learning policy for each agent."""
    
    def __init__(self, agent_name: str, learning_rate: float = 0.1, discount_factor: float = 0.9):
        self.agent_name = agent_name
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table: Dict[str, Dict[str, float]] = {}  # Q-values for different states and actions

    def get_action(self, state: str, possible_actions: List[str]) -> str:
        """Decide the action using an epsilon-greedy approach."""
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in possible_actions}

        epsilon = 0.1
        if random.random() < epsilon:
            return random.choice(possible_actions)

        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-values based on reward."""
        if next_state not in self.q_table:
            self.q_table[next_state] = {action: 0.0 for action in self.q_table[state]}
        
        max_future_q = max(self.q_table[next_state].values())
        current_q = self.q_table[state][action]
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)
        self.q_table[state][action] = new_q

class MultiAgentReinforcementLearning:
    """Handles multi-agent RL and agent selection based on relevance."""
    
    def __init__(self, agents: List[str]):
        self.agent_policies = {agent: AgentPolicy(agent) for agent in agents}

    def choose_agent(self, state: str, agent_names: List[str]) -> str:
        """Select an agent based on relevance score."""
        best_agent = random.choice(agent_names)  # Placeholder for relevance-based agent selection
        return best_agent

    def reward_agent(self, agent_name: str, state: str, action: str, reward: float, next_state: str):
        """Reward the agent for its action."""
        agent_policy = self.agent_policies[agent_name]
        agent_policy.update_q_value(state, action, reward, next_state)

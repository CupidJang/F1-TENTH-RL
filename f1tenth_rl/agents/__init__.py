from f1tenth_rl.agents.continuous import DDPGAgent, SACAgent, TD3Agent
from f1tenth_rl.agents.dqn import DQNAgent
from f1tenth_rl.agents.policy_gradient import A2CAgent, PPOAgent, REINFORCEAgent

AGENTS = {
    'dqn': DQNAgent,
    'reinforce': REINFORCEAgent,
    'a2c': A2CAgent,
    'ppo': PPOAgent,
    'ddpg': DDPGAgent,
    'td3': TD3Agent,
    'sac': SACAgent,
}


def make_agent(config, observation_space, action_space, device='cpu'):
    config = dict(config)
    return AGENTS[config.pop('name')](observation_space, action_space, device=device, **config)

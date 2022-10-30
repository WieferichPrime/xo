import random
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO

from state import check_game_status, \
    after_action_state


class BaseAgent(object):
    def __init__(self, code, model=None):
        self.code = code
        try:
            self.model = PPO.load(model)
        except BaseException as e:
            self.model = None
            print(e)

    def random_act(self, state, ava_actions):
        for action in ava_actions:
            nstate = after_action_state(state, action)
            gstatus = check_game_status(nstate[0])
            if gstatus > 0:
                if gstatus == self.code:
                    return action
        return random.choice(ava_actions)

    def act(self, state, ava_actions):
        if self.model is not None:
            action, _states = self.model.predict(state)
            while action not in ava_actions:
                action, _states = self.model.predict(state)
            nstate = after_action_state(state, action)
            gstatus = check_game_status(nstate[0])
            if gstatus > 0:
                if gstatus == self.code:
                    return action
            return action
        else:
            return self.random_act(state, ava_actions)

    def training(self, env):
        check_env(env)
        self.model = PPO("MultiInputPolicy", env, verbose=1)
        self.model.learn(total_timesteps=25000)
        self.model.save("ppo_cartpole")


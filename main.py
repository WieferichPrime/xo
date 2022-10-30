import sys

from agent import BaseAgent
from state import TicTacToeEnv, agent_by_code, next_code
from human_agent import HumanAgent


def play(vs_agent, show_number):
    env = TicTacToeEnv(show_number=show_number)
    td_agent = BaseAgent(1)
    # td_agent.training(env)
    start_code = 1
    agents = [vs_agent, td_agent]

    while True:
        env.set_start_code(start_code)
        state = env.reset()
        _, code = state.values()
        done = False
        if code == 1:
            env.render(mode='human')

        while not done:
            agent = agent_by_code(agents, code)
            human = isinstance(agent, HumanAgent)

            env.show_turn(True, code)
            ava_actions = env.available_actions()
            if human:
                action = agent.act(ava_actions)
                if action is None:
                    sys.exit()
            else:
                action = agent.act(state, ava_actions)

            state, reward, done, info = env.step(action)

            env.render(mode='human')
            if done:
                env.show_result(True, code, reward)
                break
            else:
                _, code = state.values()

        start_code = next_code(start_code)


def main():
    player = HumanAgent(2)
    play(player, False)


if __name__ == "__main__":
    main()
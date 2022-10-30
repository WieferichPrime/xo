import logging
import gym
from gym import spaces
import numpy as np

CODE_MARK_MAP = {0: ' ', 1: 'O', 2: 'X'}
NUM_LOC = 9
O_REWARD = 1
X_REWARD = -1
NO_REWARD = 0

LEFT_PAD = '  '


def tomark(code):
    return CODE_MARK_MAP[code]


def tocode(mark):
    return 1 if mark == 'O' else 2


def next_code(code):
    return 2 if code == 1 else 1


def agent_by_code(agents, code):
    for agent in agents:
        if agent.code == code:
            return agent


def after_action_state(state, action):

    board, code = state.values()
    nboard = list(board[:])
    nboard[action] = code
    nboard = tuple(nboard)
    return nboard, next_code(code)


def check_game_status(board):
    for t in [1, 2]:
        for j in range(0, 9, 3):
            if [t] * 3 == [board[i] for i in range(j, j+3)]:
                return t
        for j in range(0, 3):
            if board[j] == t and board[j+3] == t and board[j+6] == t:
                return t
        if board[0] == t and board[4] == t and board[8] == t:
            return t
        if board[2] == t and board[4] == t and board[6] == t:
            return t

    for i in range(9):
        if board[i] == 0:
            # still playing
            return -1

    # draw game
    return 0


class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, show_number=False):
        super(TicTacToeEnv, self).__init__()
        self.action_space = spaces.Discrete(NUM_LOC)
        self.observation_space = spaces.Dict({
            "board": spaces.MultiBinary(NUM_LOC),
            "code": spaces.Discrete(3)
        })
        self.start_code = 1
        self.show_number = show_number
        self.reset()

    def set_start_code(self, code):
        self.start_code = code

    def reset(self):
        self.board = np.array([0] * NUM_LOC, dtype=np.int8)
        self.code = self.start_code
        self.done = False
        return self._get_obs()

    def step(self, action):
        assert self.action_space.contains(action)

        loc = action
        if self.done:
            return self._get_obs(), 0, True, None

        reward = NO_REWARD
        self.board[loc] = self.code
        status = check_game_status(self.board)
        logging.debug("check_game_status board {} mark '{}'"
                      " status {}".format(self.board, self.code, status))
        if status >= 0:
            self.done = True
            if status in [1, 2]:
                # always called by self
                reward = O_REWARD if self.code == 1 else X_REWARD
        self.code = next_code(self.code)
        return self._get_obs(), reward, self.done, {}

    def _get_obs(self):
        return {"board": self.board, "code": self.code}

    def render(self, mode='human', close=False):
        if close:
            return
        if mode == 'human':
            self._show_board(print)
            print('')

    def _show_board(self, showfn):
        for j in range(0, 9, 3):
            def mark(i):
                return tomark(self.board[i]) if not self.show_number or\
                    self.board[i] != 0 else str(i+1)
            showfn(LEFT_PAD + '|'.join([mark(i) for i in range(j, j+3)]))
            if j < 6:
                showfn(LEFT_PAD + '-----')

    def show_turn(self, human, code):
        self._show_turn(print if human else logging.info, code)

    def _show_turn(self, showfn, code):
        showfn("{}'s turn.".format(code))

    def show_result(self, human, code, reward):
        self._show_result(print if human else logging.info, code, reward)

    def _show_result(self, showfn, mark, reward):
        status = check_game_status(self.board)
        assert status >= 0
        if status == 0:
            showfn("==== Finished: Draw ====")
        else:
            msg = "Winner is '{}'!".format(tomark(status))
            showfn("==== Finished: {} ====".format(msg))
        showfn('')

    def available_actions(self):
        return [i for i, c in enumerate(self.board) if c == 0]


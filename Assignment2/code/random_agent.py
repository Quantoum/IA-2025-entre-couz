from agent import Agent
import random
import fenix

class RandomAgent(Agent):
    def act(self, state, remaining_time):
        actions = state.actions()
        if len(actions) == 0:
            raise Exception("No action available.")
        return random.choice(actions)

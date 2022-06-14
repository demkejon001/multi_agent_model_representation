from typing import List, Union
import numpy as np


class IterativeActionAgent:
    def __init__(self, agent_id, agent_idx):
        self.id = agent_id
        self.agent_idx = agent_idx

    def reset(self):
        pass

    def action(self, state: Union[List[List[int]], None]) -> int:
        raise NotImplementedError


class MixedStrategyAgent(IterativeActionAgent):
    def __init__(self, agent_id, agent_idx, mixed_strategy):
        super().__init__(agent_id, agent_idx)
        self.num_actions = 2
        self.mixed_strategy = mixed_strategy

    def action(self, state: Union[List[List[int]], None]) -> int:
        return int(np.random.choice(self.num_actions, 1, p=self.mixed_strategy))


class PureStrategyAgent(IterativeActionAgent):
    def __init__(self, agent_id, agent_idx, agent_action):
        super().__init__(agent_id, agent_idx)
        self.agent_action = agent_action

    def action(self, state: Union[List[List[int]], None]) -> int:
        return self.agent_action


class ActionPatternAgent(IterativeActionAgent):
    def __init__(self, agent_id, agent_idx, pattern):
        super().__init__(agent_id, agent_idx)
        self.pattern = pattern
        self.pattern_idx = -1

    def reset(self):
        self.pattern_idx = -1

    def action(self, state: Union[List[List[int]], None]) -> int:
        self.pattern_idx += 1
        return self.pattern[self.pattern_idx % len(self.pattern)]


class ActionPatternAndMixedStrategyAgent(ActionPatternAgent):
    def __init__(self, agent_id, agent_idx, pattern, mixed_strategy):
        super().__init__(agent_id, agent_idx, pattern)
        self.pattern_len = len(pattern)
        self.step = -1
        self.num_actions = 2
        self.mixed_strategy = mixed_strategy
        self.follow_action_pattern = False

    def reset(self):
        super().reset()
        self.step = -1
        self.follow_action_pattern = False

    def action(self, state: Union[List[List[int]], None]) -> int:
        self.step += 1
        if (self.step % self.pattern_len) == 0:
            self.follow_action_pattern = not self.follow_action_pattern

        if self.follow_action_pattern:
            return super().action(state)
        else:
            return int(np.random.choice(self.num_actions, 1, p=self.mixed_strategy))


class PureStrategyAndMixedStrategyAgent(MixedStrategyAgent):
    def __init__(self, agent_id, agent_idx, pure_strategy_action, pure_strategy_len, mixed_strategy):
        super().__init__(agent_id, agent_idx, mixed_strategy)
        self.pure_strategy_len = pure_strategy_len
        self.pure_strategy_action = pure_strategy_action
        self.step = -1
        self.num_actions = 2
        self.follow_pure_strategy = False

    def reset(self):
        super().reset()
        self.step = -1
        self.follow_pure_strategy = False

    def action(self, state: Union[List[List[int]], None]) -> int:
        self.step += 1
        if (self.step % self.pure_strategy_len) == 0:
            self.follow_pure_strategy = not self.follow_pure_strategy

        if self.follow_pure_strategy:
            return self.pure_strategy_action
        else:
            return super().action(state)


class ReactiveAgent(IterativeActionAgent):
    def __init__(self, agent_id, agent_idx, starting_action):
        super().__init__(agent_id, agent_idx)
        self.starting_action = starting_action
        self.use_start_action = True
        self.other_agent_idx = (self.agent_idx + 1) % 2

    def reset(self):
        self.use_start_action = True

    def reactive_action(self, state: Union[List[List[int]], None]) -> int:
        raise NotImplementedError
    
    def action(self, state: Union[List[List[int]], None]) -> int:
        if self.use_start_action:
            self.use_start_action = False
            return self.starting_action
        else:
            return self.reactive_action(state)
    

class MirrorAgent(ReactiveAgent):
    def reactive_action(self, state: Union[List[List[int]], None]) -> int:
        return state[-1][self.other_agent_idx]


class ActionPatternAndMirrorAgent(ActionPatternAgent):
    def __init__(self, agent_id, agent_idx, pattern):
        super().__init__(agent_id, agent_idx, pattern)
        self.pattern_len = len(pattern)
        self.step = -1
        self.follow_action_pattern = False
        self.other_agent_idx = (self.agent_idx + 1) % 2

    def reset(self):
        super().reset()
        self.step = -1
        self.follow_action_pattern = False

    def action(self, state: Union[List[List[int]], None]) -> int:
        self.step += 1
        if (self.step % self.pattern_len) == 0:
            self.follow_action_pattern = not self.follow_action_pattern

        if self.follow_action_pattern:
            return super().action(state)
        else:
            return state[-1][self.other_agent_idx]


class PureStrategyAndMirrorAgent(PureStrategyAgent):
    def __init__(self, agent_id, agent_idx, pure_strategy_action, pure_strategy_len):
        super().__init__(agent_id, agent_idx, pure_strategy_action)
        self.pure_strategy_len = pure_strategy_len
        self.step = -1
        self.follow_pure_strategy = False
        self.other_agent_idx = (self.agent_idx + 1) % 2

    def reset(self):
        super().reset()
        self.step = -1
        self.follow_pure_strategy = False

    def action(self, state: Union[List[List[int]], None]) -> int:
        self.step += 1
        if (self.step % self.pure_strategy_len) == 0:
            self.follow_pure_strategy = not self.follow_pure_strategy

        if self.follow_pure_strategy:
            return super().action(state)
        else:
            return state[-1][self.other_agent_idx]


class BaseTriggerAgent(ReactiveAgent):
    def __init__(self,
                 agent_id,
                 agent_idx,
                 starting_action,
                 trigger_action: Union[List[int], int],
                 trigger_patience):
        super().__init__(agent_id, agent_idx, starting_action)
        self.trigger_actions = trigger_action
        if type(self.trigger_actions) is int:
            self.trigger_actions = [trigger_action]

        if trigger_patience <= 0:
            raise ValueError("trigger_patience needs to be >0")
        self.trigger_patience = trigger_patience
        self.triggered_idx = 0
        self.triggered = False

    def reset(self):
        super().reset()
        self.reset_trigger()

    def reset_trigger(self):
        self.triggered_idx = 0
        self.triggered = False

    def triggered_action(self, state: Union[List[List[int]], None]) -> int:
        raise NotImplementedError

    def untriggered_action(self, state: Union[List[List[int]], None]) -> int:
        raise NotImplementedError

    def untrigger(self):
        raise NotImplementedError

    def is_triggered(self, state: Union[List[List[int]], None]) -> bool:
        if self.triggered:
            return True

        if len(state) >= len(self.trigger_actions):
            for i in range(1, len(self.trigger_actions) + 1):
                if state[-i][self.other_agent_idx] != self.trigger_actions[-i]:
                    return False
            self.triggered_idx += 1

        if self.triggered_idx >= self.trigger_patience:
            self.triggered = True

        return self.triggered

    def reactive_action(self, state: Union[List[List[int]], None]) -> int:
        if self.is_triggered(state):
            action = self.triggered_action(state)
            self.untrigger()
            return int(action)
        else:
            return int(self.untriggered_action(state))


class GrimTriggerAgent(BaseTriggerAgent):
    def __init__(self, agent_id, agent_idx, starting_action, trigger_action):
        super().__init__(agent_id, agent_idx, starting_action, trigger_action, 1)
        self.trigger_action = trigger_action
        self.triggered = False

    def reset(self):
        super().reset()
        self.triggered = False

    def triggered_action(self, state: Union[List[List[int]], None]) -> int:
        return (self.starting_action + 1) % 2

    def untriggered_action(self, state: Union[List[List[int]], None]) -> int:
        return self.starting_action

    def untrigger(self):
        pass


class TriggerPatternAgent(BaseTriggerAgent):
    def __init__(self, agent_id, agent_idx, starting_action, trigger_action, trigger_patience, action_pattern):
        super().__init__(agent_id, agent_idx, starting_action, trigger_action, trigger_patience)
        self.trigger_action = trigger_action
        self.pattern = action_pattern
        self.pattern_idx = -1
        
    def triggered_action(self, state: Union[List[List[int]], None]) -> int:
        self.pattern_idx += 1
        return self.pattern[self.pattern_idx % len(self.pattern)]

    def untriggered_action(self, state: Union[List[List[int]], None]) -> int:
        return self.starting_action

    def reset(self):
        super().reset()
        self.pattern_idx = -1

    def untrigger(self):
        if self.pattern_idx >= (len(self.pattern) - 1):
            self.pattern_idx = -1
            self.reset_trigger()


class MixedTriggerPattern(TriggerPatternAgent):
    def __init__(self, agent_id, agent_idx, mixed_strategy, trigger_action, trigger_patience, action_pattern):
        super().__init__(agent_id, agent_idx, -1, trigger_action, trigger_patience, action_pattern)
        self.mixed_strategy = mixed_strategy
        self.num_actions = 2
        self.use_start_action = False

    def reset(self):
        super().reset()
        self.use_start_action = False

    def untriggered_action(self, state: Union[List[List[int]], None]) -> int:
        return int(np.random.choice(self.num_actions, 1, p=self.mixed_strategy))



class WSLSAgent(ReactiveAgent):
    def __init__(self, agent_id, agent_idx, starting_action, trigger_action):
        super().__init__(agent_id, agent_idx, starting_action)
        self.trigger_action = trigger_action

    def reactive_action(self, state: Union[List[List[int]], None]) -> int:
        if state[-1][self.other_agent_idx] == self.trigger_action:
            return 1 - state[-1][self.agent_idx]
        else:
            return state[-1][self.agent_idx]












import random
import pickle


class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def loadQ(self, filename):
        try:
            with open(filename + '.pickle', 'rb') as file:
                self.q = pickle.load(file)
        except Exception as e:
            print(f"Could not load the Q-table: {e}")
        print("Loaded Q-table from file: {}".format(filename+".pickle"))

    def saveQ(self, filename):
        try:
            with open(filename + '.pickle', 'wb') as file:
                pickle.dump(self.q, file)
        except Exception as e:
            print(f"Could not save the Q-table: {e}")
        print("Saved Q-table to file: {}".format(filename+".pickle"))

    def getQ(self, state, action):
        '''
        @brief returns the state, action Q value or 0.0 if the value is 
            missing
        '''
        return self.q.get((state, action), 0.0)

    def chooseAction(self, state, return_q=False):
        '''
        @brief returns a random action epsilon % of the time or the action 
            associated with the largest Q value in (1-epsilon)% of the time
        '''
        # TODO: Implement exploration vs exploitation
        #    if we need to take a random action:
        #       * return a random action
        #    else:
        #       * determine which action has the highest Q value for the state 
        #          we are in.
        #       * address edge cases - what if 2 actions have the same max Q 
        #          value?
        #       * return the action with highest Q value
        #
        # NOTE: if return_q is set to True return (action, q) instead of
        #       just action

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE 

        q_values = [self.getQ(state, a) for a in self.actions]
        maxQ = max(q_values)

        if random.random() < self.epsilon:
            # Exploration: random action
            action = random.choice(self.actions)
        else:
            # Exploitation: choose action with max Q value
            count = q_values.count(maxQ)
            if count > 1:
                # In case there're several 'max Q values', select a random one among them
                best_actions = [i for i in range(len(self.actions)) if q_values[i] == maxQ]
                i = random.choice(best_actions)
            else:
                i = q_values.index(maxQ)

            action = self.actions[i]

        return (action, maxQ) if return_q else action

    def learn(self, state1, action1, reward, state2):
        '''
        @brief updates the Q(state,value) dictionary using the bellman update
            equation
        '''
        # TODO: Implement the Bellman update function:
        #     Q(s1, a1) += alpha * [reward(s1,a1) + gamma* max(Q(s2)) - Q(s1,a1)]
        # 
        # NOTE: address edge cases: i.e. 
        # 
        # Find Q for current (state1, action1)
        # Address edge cases what do we want to do if the [state, action]
        #       is not in our dictionary?
        # Find max(Q) for state2
        # Update Q for (state1, action1) (use discount factor gamma for future 
        #   rewards)

        # THE NEXT LINES NEED TO BE MODIFIED TO MATCH THE REQUIREMENTS ABOVE

        # Get the max Q value for the next state
        max_q_new_state = max([self.getQ(state2, a) for a in self.actions])
        # Get the current Q value for the current state-action pair
        current_q_value = self.getQ(state1, action1)
        # Calculate the updated Q value
        updated_q_value = current_q_value + self.alpha * (reward + self.gamma * max_q_new_state - current_q_value)
        # Update the Q-table
        self.q[(state1, action1)] = updated_q_value

    def save_best_reward(self, filename, reward):
        with open(filename + '_reward.pickle', 'wb') as f:
            pickle.dump(reward, f)

    def load_best_reward(self, filename):
        try:
            with open(filename + '_reward.pickle', 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return -float('inf')  # Return negative infinity if file not found

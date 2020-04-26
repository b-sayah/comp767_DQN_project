class PER:
    # prioritizing using a replay memory of 100000 transitions
    n_transitions = 10 ** 5
    # factor that determines how much prioritization is applied
    alpha = 0.6

    # TODO add alpha, beta, n_transition and minibatch, we can play with them
    def __init__(self):

        self.priorities = np.zeros(self.n_transitions, np.float32)
        self.i = 0  # index
        self.rep_memory = list()

    def full_memory(self, memory_sz):
        """test if the replay buffer is full"""
        return (memory_sz >= self.n_transitions)

    def highest_priority(self, memory_sz):
        """get the hightest priority"""
        if memory_sz == 0:
            return 1
        elif memory_sz >= 1:
            return self.priorities.max()

    def store(self, s, a, r, s_prime, done):
        """Store or add a transition to the Replay Memory"""

        s, s_prime = np.asarray([s]), np.asarray([s_prime])

        memory_sz = len(self.rep_memory)
        rep_memory_full = self.full_memory(memory_sz)

        if rep_memory_full == True:
            self.rep_memory[self.i] = (s, a, r, s_prime, done)
        else:
            self.rep_memory.append((s, a, r, s_prime, done))

        self.priorities[self.i] = self.highest_priority(memory_sz)

        self.i += 1
        self.i %= self.n_transitions

    def replay(self, n_sample, beta):
        """replaying transitions following probabilty of sampling transition i
        defined in equation (1) of the Prioritzed Experience Replay paper """
        memory_sz = len(self.rep_memory)

        if memory_sz != self.n_transitions:
            priorities = self.priorities[:self.i]
        else:
            priorities = self.priorities

        # Probabilities of sampling transitions
        probabilities = priorities ** self.alpha
        probabilities = probabilities / probabilities.sum()

        idx = np.random.choice(a=memory_sz,
                               size=n_sample,
                               replace=True,
                               p=probabilities)

        # compute importance-sampling weights
        # from Algorithm 1 of PER paper
        W = 1 / (probabilities[idx] * memory_sz) ** beta
        W = W / W.max()

        S, A, R, S_prime, Done = zip(*(self.rep_memory[i] for i in idx))
        S, S_prime = map(np.concatenate, (S, S_prime))

        return S, A, R, S_prime, Done, W, idx,
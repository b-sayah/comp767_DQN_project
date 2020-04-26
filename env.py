class Env(gym.Wrapper):
    def __init__(self, game_name):
        env = gym.make(f"{game_name}NoFrameskip-v4")
        super().__init__(env)
        self.rep_memory = np.zeros((2, *env.observation_space.shape), np.uint8)
        self.lives = 0
        self.done = True
        self.observation_space = Box(0, 1, (1, 84, 84), np.uint8)

    def _reset(self, **kwargs):
        if self.done:
            self.env.reset(**kwargs)
            # Test evaluation method (<= 30 random no-ops)
            for _ in range(self.unwrapped.np_random.randint(1, 31)):
                s, _, done, _ = self.env.step(0)
                if done:
                    s = self.env.reset(**kwargs)
        else:
            done = self.done  # backup b/c step() overrides it
            s, _, _, _ = self.env.step(0)  # re-spawn back to life
            self.done = done

    def reset(self, **kwargs):
        self._reset(**kwargs)
        if "FIRE" in self.unwrapped.get_action_meanings():
            s, _, done, _ = self.step(1)  # Activate firing in environment
            if done:
                self._reset(**kwargs)
            s, _, done, _ = self.step(2)
            if done:
                self._reset(**kwargs)
        return s

    def step(self, a):
        reward = 0.0
        for i in range(4):
            s, r, done, info = self.env.step(a)
            if i in (2, 3):
                self.rep_memory[i - 2] = s
            reward += r
            if done:
                break
        s = self.rep_memory.max(0)

        self.done = done
        lives = self.env.unwrapped.ale.lives()
        done = 0 < lives < self.lives or done
        self.lives = lives
        r = np.sign(reward)  # clip to [-1, 1]
        return self.observation(s), r, done, info

    def observation(self, s):
        """As in Nature paper"""
        s = cv2.resize(cv2.cvtColor(s, cv2.COLOR_RGB2GRAY),
                       (84, 84), interpolation=cv2.INTER_AREA)[:, :, None]
        s = np.swapaxes(s, 2, 0)  # to feed into DQN
        return s


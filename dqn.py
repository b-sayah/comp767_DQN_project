class DQN(nn.Module):

    """model architecture follows the paper
    Human-level control through deep reinforcement learning
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf"""

    def __init__(self, n_actions):
        super().__init__()
        self.n_actions = n_actions
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.net(x)
        return x

    def action_selection(self, s, epsilon):

        if epsilon >= np.random.random():
            a = np.random.choice(self.n_actions)
            return a
        # else:? TODO check
        with torch.no_grad():
            s = torch.tensor([s], dtype=torch.float32, device=DEVICE)
            Q_qnet = self(s)
            a = Q_qnet.argmax(1).item()
            return a
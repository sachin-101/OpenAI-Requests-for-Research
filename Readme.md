### OpenAI Requests For Research 2.0

OpenAI has released a new batch of seven unsolved problems which have come up in the course of their research at OpenAI. These problems are a fun and meaningful way for new people to enter the field, as well as for practitioners to hone their skills. Many will require inventing new ideas.

### Warmups
- [ ] Train an LSTM to solve xor problem.
- [x] Implement a clone of classic Sanke Game and solve it using RL algorithm.

### Main Problem statements

- [ ] Slitherin'. : Implement and solve a multiplayer clone of classic Snake game.  (***Work Going On***)
- [ ] Parameter Averaging in Distributed RL 
- [ ] Transfer learning between different games via Generative Models
- [ ] Transformers with Linear Attention
- [ ] Learned Data Augmentation
- [ ] Regularization in Reinforcement Learning
- [ ] Automated Solutions of Olympiad Inequality Problems

### Training Agent

![Training Agent](https://github.com/sachin-101/OpenAI-Requests-for-Research/blob/master/Snake%20Game/Video/Snake-Game-Training.gif)

- Using **Double DQN**(RL algo) to train the AI agent.
- Using **Curriculum Learning** approach to train the agent. Basically, keeping the playing area small at the beginning of training.
And increasing the size of playing area as the training progresses.
- This helps to solve the problem of **Sparse Reward**, cause if the agent starts in a very big playing area, it becomes very difficult for it to find positive reward in th environment, thus disabling the agent from learning the optimal solution.
- Small playing area ensures that the agent bumps into the food often, even while performing random actions(epsilon close to 1).

### Learned Agent

![Learned Agent](https://github.com/sachin-101/OpenAI-Requests-for-Research/blob/master/Snake%20Game/Video/Trained-Agent.gif)

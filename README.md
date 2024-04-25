# ai-rl-agent
Reinforcement Learning Agent Team Project for Artificial Intelligence
1. REINFORCE for control with linear function approximation
   Run using: python pacman.py -p ReinforceMCPGAgent -q -a extractor=CustomExtractor,alpha=0.3,epsilon=0.05 -x 200 -n 210 -l mediumClassic
2. Actor Critic for control with linear function approximation
   Run using: python pacman.py -p ActorCriticAgent -q -a extractor=CustomExtractor,alpha=0.3,epsilon=0.05 -x 200 -n 210 -l mediumClassic
3. QLearning for control with linear function approximation
   Run using: python pacman.py -p ApproximateQAgent -q -a extractor=SimpleExtractor,alpha=0.3,epsilon=0.05 -x 200 -n 210 -l mediumClassic

To run plots:

For varying alpha values:
1. REINFORCE: python plot_alpha_REINFORCE.py
2. ActorCritic: python plot_alpha_ActorCritic.py
3. QLearning: python plot_alpha_ApproxQLearning.py

For a comparision between all 3 algorithms:
1. mediumClassic layout:python plot_diff_med.py
1. smallClassic layout:python plot_diff_small.py


import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import threading

# Define the fixed alpha value
alpha = 0.1

# Define the number of iterations
num_iterations = 20

# Define the agent types to iterate over
agent_types = ["ReinforceMCPGAgent", "ActorCriticAgent", "ApproximateQAgent"]
extractors = ["CustomExtractor", "CustomExtractor", "SimpleExtractor"]

all_average_scores = []

def run_experiment(agent_type, extractor):
    average_scores = []
    for _ in range(num_iterations):
        # Construct the command with the current agent type and fixed alpha value
        command = [
            "python",
            "pacman.py",
            "-p", agent_type,
            "-a", f"extractor={extractor},alpha={alpha},epsilon=0.05",
            "-x", "200",
            "-n", "200",
            "-l", "smallClassic"
        ]
        # Run the command with subprocess
        result = subprocess.run(command, stdout=subprocess.PIPE)
        output_lines = result.stdout.decode('utf-8').split('\n')
        
        episodes = []
        scores = []

        for i, line in enumerate(output_lines):
            match = re.match(r"ep (\d+)", line)
            if match:
                episode = int(match.group(1))
                if i + 1 < len(output_lines):
                    try:
                        score = float(output_lines[i + 1])
                        episodes.append(episode)
                        scores.append(score)
                    except ValueError:
                        pass
        
        # Append the scores for this iteration
        average_scores.append(scores)
    
    # Calculate the average score for each episode across all iterations
    average_scores = np.mean(np.array(average_scores), axis=0)
    
    # Store the average scores for this agent type
    all_average_scores.append((agent_type, average_scores))

# Create threads to run experiments for each agent type
threads = []
for agent_type, extractor in zip(agent_types, extractors):
    thread = threading.Thread(target=run_experiment, args=(agent_type, extractor,))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Plot the combined graph
plt.figure(figsize=(20, 6))  # Adjust figure size if necessary
for agent_type, average_scores in all_average_scores:
    plt.plot(range(len(average_scores)), average_scores, marker=None)

plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.title('Average Score vs Episode')
#plt.grid(True)
plt.legend(agent_types)
plt.yticks(range(-1000, 2000, 500))

plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.show()

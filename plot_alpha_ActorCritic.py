import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import threading

# Define the range of alpha values you want to iterate over
alpha_values = [0.01, 0.05, 0.1, 0.2, 0.5]

# Define the number of iterations
num_iterations = 20

all_average_scores = []

def run_experiment(alpha):
    average_scores = []
    for _ in range(num_iterations):
        # Construct the command with the current alpha value
        command = [
            "python",
            "pacman.py",
            "-p", "ActorCriticAgent",
            "-a", f"extractor=CustomExtractor,alpha={alpha},epsilon=0.05",
            "-x", "400",
            "-n", "400",
            "-l", "mediumClassic"
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
    
    # Store the average scores for this alpha value
    all_average_scores.append((alpha, average_scores))

# Create threads to run experiments for each alpha value
threads = []
for alpha in alpha_values:
    thread = threading.Thread(target=run_experiment, args=(alpha,))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

# Sort the results by alpha value
all_average_scores.sort(key=lambda x: x[0])

# Plot the combined graph
plt.figure(figsize=(20, 6))  # Adjust figure size if necessary
for alpha, average_scores in all_average_scores:
    plt.plot(range(len(average_scores)), average_scores, marker=None)

plt.xlabel('Episode')
plt.ylabel('Average Score')
plt.title('Average Score vs Episode')
#plt.grid(True)
plt.legend(['Alpha = ' + str(alpha) for alpha, _ in all_average_scores])
plt.yticks(range(-1000, 2000, 500))

plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

plt.show()

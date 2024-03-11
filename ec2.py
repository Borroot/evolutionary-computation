import numpy as np
import matplotlib.pyplot as plt
import copy
import os

fig_dir = 'results2/'

def iteration(x, l, mu, elitism, n_parallel):

	x_m = copy.copy(x)
	p = np.random.rand(n_parallel, l)
	# If samples are smaller than the threshold, flip bits
	mask = p < mu
	x_m[mask] = 1-x_m[mask]

	# Compute fitness
	x_m_value = np.sum(x_m, axis=1)
	x_value = np.sum(x, axis=1)
	
	if elitism:
		# Get single arrays of the best fitnesses and binary
		# representations between the child and parent, for each
		# trial
		child_fitter = x_m_value > x_value
		x_value[child_fitter] = x_m_value[child_fitter]
		x[child_fitter,:] = x_m[child_fitter, :]
		return x, x_value
	else:
		return x_m, x_m_value


def main(l, mu, n_generations, elitism, n_parallel):

	x = np.random.choice((0,1), (n_parallel,l))
	fitnesses = np.zeros((n_parallel, n_generations))
	for g in range(n_generations):
		x, x_fitness = iteration(x, l, mu, elitism, n_parallel)
		fitnesses[:,g] = x_fitness 

	return fitnesses 

def plot_single_trials(l, mu, n_generations):

	# Plot 1, where we only replace the parent if the child is better
	fitnesses_elitist = main(l, mu, n_generations, 
		elitism=True, n_parallel=1)
	plt.figure()
	plt.plot(fitnesses_elitist[0], label='(+) elitism', color='tab:blue')
	plt.xlabel('Generation')
	plt.ylabel('Entry sum')
	plt.axhline(100, color='tab:red', linestyle='--', label='Target')
	plt.legend()
	plt.xlim([0, 1500])
	plt.savefig(os.path.join(fig_dir, 'part_1.png'), bbox_inches="tight", dpi=300)
	plt.show()

	# Plot 2, where we always replace the parent
	fitnesses_non_elitist = main(l, mu, n_generations, 
		elitism=False, n_parallel=1)
	plt.figure()
	plt.plot(fitnesses_elitist[0], label='(+) elitism', color='tab:blue')
	plt.plot(fitnesses_non_elitist[0], label='(-) elitism', color='tab:orange')
	plt.xlabel('Generation')
	plt.ylabel('Entry sum')
	plt.axhline(100, color='tab:red', label='Target', linestyle='--')
	plt.legend()
	plt.xlim([0, 1500])
	plt.savefig(os.path.join(fig_dir, 'part_2.png'), bbox_inches="tight", dpi=300)
	plt.show()

def plot_multiple_trials(l, mu, n_generations, n_trials):

	fitnesses_elitist = main(l, mu, n_generations, 
		elitism=True, n_parallel=n_trials)
	fitnesses_non_elitist = main(l, mu, n_generations, 
		elitism=False, n_parallel=n_trials)

	# Standard deviations
	std_lower_elitist = np.mean(fitnesses_elitist, axis=0) - np.std(fitnesses_elitist, axis=0)
	std_upper_elitist = np.mean(fitnesses_elitist, axis=0) + np.std(fitnesses_elitist, axis=0)
	std_lower_non_elitist = np.mean(fitnesses_non_elitist, axis=0) - \
		np.std(fitnesses_non_elitist, axis=0)
	std_upper_non_elitist = np.mean(fitnesses_non_elitist, axis=0) + \
		np.std(fitnesses_non_elitist, axis=0)


	plt.figure()

	plt.plot(np.mean(fitnesses_elitist, axis=0), label='(+) elitism', color='tab:blue')
	plt.fill_between(np.arange(0,n_generations), 
		std_lower_elitist, std_upper_elitist, alpha=0.15, color='tab:blue', 
		label='(+) elitism $\\sigma$ range')

	plt.plot(np.mean(fitnesses_non_elitist, axis=0), label='(-) elitism', color='tab:orange')
	plt.fill_between(np.arange(0,n_generations), 
		std_lower_non_elitist, std_upper_non_elitist, alpha=0.15, color='tab:orange', 
		label='(-) elitism $\\sigma$ range')

	plt.xlabel('Generation')
	plt.ylabel('Entry sum')
	plt.axhline(100, color='tab:red', label='Target', linestyle='--')
	plt.legend()
	plt.xlim([0, 1500])
	plt.savefig(os.path.join(fig_dir, 'part_3.png'), bbox_inches="tight", dpi=300)
	plt.show()


if __name__ == '__main__':

	# Comparing individual trials
	l = 100
	mu = 1/l
	n_generations = 1500
	plot_single_trials(l, mu, n_generations)

	# Comparing multiple trials
	n_trials = 1000
	plot_multiple_trials(l, mu, n_generations, n_trials)



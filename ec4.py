import numpy as np
import matplotlib.pyplot as plt
import os
import copy
from tqdm import tqdm
from itertools import combinations

fig_dir = 'results4/'


def ea(N, K, mu, mu_cross, coordinates, n_iter, n_trials, swap_frac, local_search):
	# Implements the evolutionary algorithm covered in the lecture

	n_locations = coordinates.shape[0]

	fitnesses_over_trials = np.zeros((n_trials, n_iter+1, N))
	solutions_over_trials = np.zeros((n_trials, n_iter+1, N, n_locations))

	for k in tqdm(range(n_trials)):
		# Generate a population of N candidate solutions
		current_solution_pool = np.zeros((N, n_locations))
		for i in range(N):
			current_solution_pool[i] = np.random.choice(
				np.arange(0, n_locations), n_locations, replace=False)

		current_solution_pool = current_solution_pool.astype(int)

		if local_search:
			# Performing local search for each candidate solution
			for solution in current_solution_pool:
				solution = opt_swap(solution, coordinates, swap_frac)

		# For storing everything throughout the trial
		fitnesses = np.zeros((n_iter+1, N))
		solutions = np.zeros((n_iter+1, N, n_locations))

		# Evaluate the current solutions
		for i, solution in enumerate(current_solution_pool):
			fitnesses[0, i] = get_fitness(solution, coordinates)
			solutions[0, i, :] = solution

		for i in range(n_iter):
			# Perform tournament selection to create a new generation
			new_solutions = np.zeros_like(current_solution_pool)
			for j in range(0, N, 2):
				# Select parents using tournament selection
				parent_1 = tournament_selection(current_solution_pool, K, coordinates)
				parent_2 = tournament_selection(current_solution_pool, K, coordinates)

				# Crossover and mutation
				child_1, child_2 = order_crossover(parent_1, parent_2, mu_cross)
				child_1 = mutation(child_1, mu)
				child_2 = mutation(child_2, mu)

				if local_search:
					# Performing local search to potentially improve
					# the children
					child_1 = opt_swap(child_1, coordinates, swap_frac)
					child_2 = opt_swap(child_2, coordinates, swap_frac)

				new_solutions[j] = child_1
				new_solutions[j+1] = child_2

				fitnesses[i+1, j] = get_fitness(child_1, coordinates)
				fitnesses[i+1, j+1] = get_fitness(child_2, coordinates)

			# Update the generation, keeping only the fittest
			current_and_previous_fitnesses = \
				np.concatenate((fitnesses[i], fitnesses[i+1]), axis=0)
			current_and_previous_solutions = \
				np.concatenate((current_solution_pool, new_solutions), axis=0)

			fittest_solutions = np.flip(
				np.argsort(current_and_previous_fitnesses))[:N]
			new_solutions = current_and_previous_solutions[fittest_solutions,:]

			# Changing the variable because we reuse it in the next iteration
			current_solution_pool = new_solutions
			solutions[i+1] = current_solution_pool

		fitnesses_over_trials[k] = fitnesses
		solutions_over_trials[k] = solutions

	return fitnesses_over_trials, solutions_over_trials

def opt_swap(permutation, coordinates, swap_frac):
	# Performs a 2-opt local search step on a given 
	# candidate solution

	# Generate all possible combinations of indices to swap
	# except for redundant self-swaps 
	swap_combinations = np.array(
		list(combinations(np.arange(0, len(permutation)), 2)))

	# Sample a fraction of these randomly (doing them all takes
	# way too long)
	samples = np.random.choice(
		np.arange(len(swap_combinations)), 
		int(len(swap_combinations)*swap_frac),
		replace=False)

	swap_combinations = swap_combinations[samples]

	# Evaluate the fitness, and iterate through each possible
	# swap combination to see if the fitness improves.
	fitness = get_fitness(permutation, coordinates)
	for swap in swap_combinations:
		permutation[swap] = permutation[np.flip(swap)]
		swap_fitness = get_fitness(permutation, coordinates)
		# If better than previous, keep permutation and 
		# update the criterion
		if swap_fitness > fitness:
			fitness = swap_fitness
		# Otherwise, swap back
		else:
			permutation[swap] = permutation[np.flip(swap)]

	return permutation

def get_fitness(permutation, coordinates):
	# Generate an array of the coordinate indices between which
	# the distances have to be evaluated 
	from_idx = permutation
	to_idx = np.zeros_like(permutation)
	# "Shift this over" with a wrap-around of 1
	to_idx[:-1] = permutation[1:]
	to_idx[-1] = permutation[0]

	# Calculate euclidean distances between the pairs of points
	dx = (coordinates[from_idx][:,0] - coordinates[to_idx][:,0])**2
	dy = (coordinates[from_idx][:,1] - coordinates[to_idx][:,1])**2
	distances = np.sqrt(dx + dy)

	return 1/np.sum(distances)

def order_crossover(p1, p2, mu_cross):
	# Implements order crossover as covered in the lecture, with
	# probability mu_cross.
	p = np.random.rand()
	if p > mu_cross: # Do nothing!
		return p1, p2

	# Sample two crossover points which are different. Also want to avoid
	# situation where crossover indices are the first and last indices of 
	# the permutation, as this doesn't actually create new offspring.
	crossovers = [0, len(p1)-1]
	while crossovers == [0, len(p1)-1]:
		# Sample two points without replacement
		crossovers = np.sort(
			np.random.choice(
				np.arange(0, len(p1)), 2, replace=False))
		crossovers = list(crossovers)

	# Finding the complements in the opposite parents
	cut_1, cut_2 = crossovers

	p1_complement = np.setdiff1d(copy.copy(p2), p1[cut_1:cut_2+1])
	p2_complement = np.setdiff1d(copy.copy(p1), p2[cut_1:cut_2+1])	

	# Defining the children, with -1 just for easy debugging
	p1_child = -1*np.ones_like(p1)
	p2_child = -1*np.ones_like(p2)

	# Putting the middle segments back in
	p1_child[cut_1:cut_2+1] = p1[cut_1:cut_2+1]
	p2_child[cut_1:cut_2+1] = p2[cut_1:cut_2+1]

	# Filling in the right empty space
	n_right = len(p1_child)-cut_2-1


	p1_child[cut_2+1:] = p1_complement[:n_right]
	p2_child[cut_2+1:] = p2_complement[:n_right]

	# Filling in the left empty space
	p1_child[:cut_1] = p1_complement[n_right:]
	p2_child[:cut_1] = p2_complement[n_right:]

	return p1_child, p2_child

def mutation(permutation, mu):
	# Performs random mutation with probability mu
	p = np.random.rand(len(permutation), len(permutation))
	np.fill_diagonal(p, 1.0) # avoid redundant swaps

	# The indices which should be swapped
	to_swap = np.argwhere(p < mu)

	# Perform the swaps in order
	for swap in to_swap:
		permutation[swap] = permutation[np.flip(swap)]

	return permutation

def tournament_selection(parents, K, coordinates):
	# Performs tournament selection given a pool of parents and 
	# value of K

	selected_parents = np.random.choice(
		np.arange(0, len(parents)), K, replace=False)

	selected = parents[selected_parents]

	# Compute fitnessess for selected parents
	fitnesses = np.zeros(K)
	for i in range(K):
		fitnesses[i] = get_fitness(selected[i], coordinates)

	# Return the fittest parent
	return selected[np.argmax(fitnesses)]

def visualize_output(fitnesses_over_trials_e, solutions_over_trials_e,
	fitnesses_over_trials_m, solutions_over_trials_m,
	 n_iter, coordinates, filename):

	# Mean fitness across generations
	mean_fitnesses_e = np.mean(fitnesses_over_trials_e, axis=(0,2))
	st_dev_e = np.mean(np.std(fitnesses_over_trials_e, axis=2), axis=0)
	lower_e = mean_fitnesses_e - st_dev_e
	upper_e = mean_fitnesses_e + st_dev_e

	mean_fitnesses_m = np.mean(fitnesses_over_trials_m, axis=(0,2))	
	st_dev_m = np.mean(np.std(fitnesses_over_trials_m, axis=2), axis=0)
	lower_m = mean_fitnesses_m - st_dev_m
	upper_m = mean_fitnesses_m + st_dev_m


	plt.figure()
	plt.plot(mean_fitnesses_e, color='tab:blue', label='Evolutionary')
	plt.fill_between(np.arange(0, n_iter+1), 
		lower_e, upper_e, alpha=0.15, color='tab:blue', 
		label='Mean evolutionary $\\sigma$ range')


	plt.plot(mean_fitnesses_m, color='tab:orange', label='Memetic')
	plt.fill_between(np.arange(0, n_iter+1), 
	lower_m, upper_m, alpha=0.15, color='tab:orange', 
	label='Mean memetic $\\sigma$ range')	

	plt.xlim([0, n_iter])
	plt.xlabel('Generation')
	plt.ylabel('Mean fitness')
	plt.legend()
	plt.savefig(os.path.join(fig_dir, f'{filename}_fitness.png'), bbox_inches="tight", dpi=300)
	plt.show()


	# Mean shortest distance across generations
	best_fitnesses_e = np.max(fitnesses_over_trials_e, axis=2)
	best_fitnesses_m = np.max(fitnesses_over_trials_m, axis=2)

	mean_distances_e = 1/np.mean(best_fitnesses_e, axis=0)
	st_dev_e = np.std(mean_distances_e, axis=0)
	lower_e = mean_distances_e - st_dev_e
	upper_e = mean_distances_e + st_dev_e

	mean_distances_m = 1/np.mean(best_fitnesses_m, axis=0)
	st_dev_m = np.std(mean_distances_m, axis=0)
	lower_m = mean_distances_m - st_dev_m
	upper_m = mean_distances_m + st_dev_m	


	plt.plot(mean_distances_e, color='tab:blue', label='Evolutionary')
	plt.fill_between(np.arange(0, n_iter+1), 
		lower_e, upper_e, alpha=0.15, color='tab:blue', 
		label='Mean evolutionary $\\sigma$ range')

	plt.plot(mean_distances_m, color='tab:orange', label='Memetic')
	plt.fill_between(np.arange(0, n_iter+1), 
		lower_m, upper_m, alpha=0.15, color='tab:orange', 
		label='Mean memetic $\\sigma$ range')

	plt.xlim([0, n_iter])
	plt.xlabel('Generation')
	plt.ylabel('Mean shortest distance traveled')
	plt.legend()
	plt.savefig(os.path.join(fig_dir, f'{filename}_distance.png'), bbox_inches="tight", dpi=300)
	plt.show()

	# Visualizing the best solution overall for evolutionary algorithm
	best_solution_idx = \
		np.unravel_index(
			fitnesses_over_trials_e.argmax(), fitnesses_over_trials_e.shape)

	best_solution = solutions_over_trials_e[best_solution_idx[0],
											best_solution_idx[1],
											best_solution_idx[2], :].astype(int)
	path = coordinates[best_solution]
	path = np.concatenate((path, path[None, 0,:]), axis=0)
	plt.plot(path[:,0], path[:,1])
	plt.scatter(coordinates[:,0],coordinates[:,1], color='tab:red')
	plt.xlabel('X position')
	plt.ylabel('Y position')
	plt.savefig(os.path.join(fig_dir, f'{filename}_e_graph.png'), bbox_inches="tight", dpi=300)
	plt.show()


	# Visualizing the best solution overall for memetic algorithm
	best_solution_idx = \
		np.unravel_index(
			fitnesses_over_trials_m.argmax(), fitnesses_over_trials_m.shape)

	best_solution = solutions_over_trials_m[best_solution_idx[0],
											best_solution_idx[1],
											best_solution_idx[2], :].astype(int)
	path = coordinates[best_solution]
	path = np.concatenate((path, path[None, 0,:]), axis=0)
	plt.plot(path[:,0], path[:,1])
	plt.scatter(coordinates[:,0],coordinates[:,1], color='tab:red')
	plt.xlabel('X position')
	plt.ylabel('Y position')
	plt.savefig(os.path.join(fig_dir, f'{filename}_m_graph.png'), bbox_inches="tight", dpi=300)
	plt.show()

if __name__ == '__main__':

	n_iter = 300
	n_trials = 10
	N = 250
	K = 2
	mu = 0.0005
	mu_cross = 0.1

	# Evolutionary algorithm on example problem
	coordinates = np.loadtxt('file-tsp.txt')
	fitnesses_over_trials, solutions_over_trials = \
		ea(N=N, K=K, mu=mu, mu_cross=mu_cross, coordinates=coordinates,
			n_iter = n_iter, n_trials = n_trials, swap_frac=0, local_search=False)
	np.save(os.path.join(fig_dir, 'fitnesses_over_trials_ea_50'), fitnesses_over_trials)
	np.save(os.path.join(fig_dir, 'solutions_over_trials_ea_50'), solutions_over_trials)

	fitnesses_over_trials_e = np.load(os.path.join(fig_dir, 'fitnesses_over_trials_ea_50.npy'))
	solutions_over_trials_e = np.load(os.path.join(fig_dir, 'solutions_over_trials_ea_50.npy'))

	# Memetic algorithm on example problem
	fitnesses_over_trials, solutions_over_trials = \
		ea(N=N, K=K, mu=mu, mu_cross=mu_cross, coordinates=coordinates,
			n_iter = n_iter, n_trials = n_trials, swap_frac=0.2, local_search=True)
	np.save(os.path.join(fig_dir, 'fitnesses_over_trials_ma_50'), fitnesses_over_trials)
	np.save(os.path.join(fig_dir, 'solutions_over_trials_ma_50'), solutions_over_trials)

	fitnesses_over_trials_m = np.load(os.path.join(fig_dir, 'fitnesses_over_trials_ma_50.npy'))
	solutions_over_trials_m = np.load(os.path.join(fig_dir, 'solutions_over_trials_ma_50.npy'))


	visualize_output(fitnesses_over_trials_e, solutions_over_trials_e, 
		fitnesses_over_trials_m, solutions_over_trials_m,
		n_iter, coordinates, 'example_problem')


	# Evolutionary algorithm on other small external problem
	with open("ulysses16.tsp", "r") as file:
		lines = [line.rstrip('\n') for line in file.readlines()]
	lines = lines[7:-2]
	array_2d = np.array([[float(num) for num in line.split()] for line in lines])
	coordinates = array_2d[:,1:]
	
	fitnesses_over_trials, solutions_over_trials = \
		ea(N=N, K=K, mu=mu, mu_cross=mu_cross, coordinates=coordinates,
			n_iter = n_iter, n_trials = n_trials, swap_frac=0, local_search=False)

	np.save(os.path.join(fig_dir, 'fitnesses_over_trials_ea_16'), fitnesses_over_trials)
	np.save(os.path.join(fig_dir, 'solutions_over_trials_ea_16'), solutions_over_trials)

	fitnesses_over_trials_e = np.load(os.path.join(fig_dir, 'fitnesses_over_trials_ea_16.npy'))
	solutions_over_trials_e = np.load(os.path.join(fig_dir, 'solutions_over_trials_ea_16.npy'))


	# Memetic algorithm on other small external problem
	
	fitnesses_over_trials, solutions_over_trials = \
		ea(N=N, K=K, mu=mu, mu_cross=mu_cross, coordinates=coordinates,
			n_iter = n_iter, n_trials = n_trials, swap_frac=0.2, local_search=True)

	np.save(os.path.join(fig_dir, 'fitnesses_over_trials_ma_16'), fitnesses_over_trials)
	np.save(os.path.join(fig_dir, 'solutions_over_trials_ma_16'), solutions_over_trials)

	fitnesses_over_trials_m = np.load(os.path.join(fig_dir, 'fitnesses_over_trials_ma_16.npy'))
	solutions_over_trials_m = np.load(os.path.join(fig_dir, 'solutions_over_trials_ma_16.npy'))
	
	visualize_output(fitnesses_over_trials_e, solutions_over_trials_e, 
		fitnesses_over_trials_m, solutions_over_trials_m,
		n_iter, coordinates, 'external_problem')










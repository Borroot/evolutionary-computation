import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import string


def init_population(alphabet, entity_size, population_size):
    """Initialize a random population."""
    return np.random.choice(alphabet, size=(population_size, entity_size))


def target_found(target, population):
    """Return whether the target is present in the given population."""
    return any(np.array_equal(individual, target) for individual in population)


def fitness_population(target, population):
    """Return the fitness, i.e. fraction of correct letters, for each individual."""
    f = lambda individual, target: np.mean(individual == target)
    return np.array([f(individual, target) for individual in population])


def init_tournament(tournament_size, population_size):
    """Return a list of (N / 2, 2, K) representing all the N / 2 tournaments."""
    return np.random.randint(population_size, size=(population_size // 2, 2, tournament_size))


def battle(parent_candidates, fitnesses, population):
    """Return the parent with the highest fitness."""
    return population[parent_candidates[np.argmax(fitnesses[parent_candidates])]]


def crossover(parent1, parent2):
    """Return the two offspring after crossover at a random split point occured."""
    crossover_point = np.random.randint(1, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2


def mutate(individual, mutation_prob, alphabet):
    """Mutate the given individual using the mutation probability given."""
    mutate_indices = np.array(np.random.rand(len(individual)) < mutation_prob)
    individual[mutate_indices] = np.random.choice(alphabet, size=sum(mutate_indices))


def tournament(tournament_size, population, population_size, mutation_prob, target, alphabet):
    """Run a tournament and return the new population."""
    fitnesses = fitness_population(target, population)
    tournament_rounds = init_tournament(tournament_size, population_size)

    new_population = []

    for index, (parent1_candidates, parent2_candidates) in enumerate(tournament_rounds):
        parent1 = battle(parent1_candidates, fitnesses, population)
        parent2 = battle(parent2_candidates, fitnesses, population)

        child1, child2 = crossover(parent1, parent2)
        mutate(child1, mutation_prob, alphabet)
        mutate(child2, mutation_prob, alphabet)
        new_population.extend([child1, child2])

    return np.array(new_population)


def run_genetic_search(
    alphabet, tournament_size, entity_size, target, mutation_prob, population_size, max_iter
):
    """Run the genetic string search algorithm."""
    population = init_population(alphabet, entity_size, population_size)
    generation = 0

    while generation < max_iter:
        population = tournament(
            tournament_size, population, population_size, mutation_prob, target, alphabet
        )

        if target_found(target, population):
            return generation, True

        generation += 1

    return generation, False


def plot(results, mutation_probs, max_iter):
    for index, mutation_prob in enumerate(mutation_probs):
        converged = results[index, results[index, :, 1] == True][:, 0]
        limit = results[index, results[index, :, 1] == False][:, 0]

        noise_scale = 0.01

        noise_converged = noise_scale * (np.random.rand(len(converged)) - 0.5)
        noise_limit = noise_scale * (np.random.rand(len(limit)) - 0.5)

        plt.vlines(mutation_prob, 0, max_iter, linestyles='--')

        plt.scatter(noise_converged + [mutation_prob] * len(converged), converged, color='darkgreen', label='converged')
        plt.scatter(noise_limit + [mutation_prob] * len(limit), limit, color='darkred', label='max iter')

    plt.ylabel('#generations')
    plt.xlabel('$\mu$')

    plt.savefig('results3/1.png')
    plt.show()


def main():
    np.random.seed(0)  # for reproducability

    # set experiment search parameters
    alphabet = np.array(list(string.ascii_letters))
    tournament_size = 2

    entity_size = 15
    target = np.random.choice(alphabet, size=entity_size)

    population_size = 200

    # set experiment hyperparameters
    mutation_probs = [0, 1 / entity_size, 3 / entity_size]

    max_iter = 100
    repetitions = 50

    # run the experiments
    results = np.array([
        [
            run_genetic_search(
                alphabet, tournament_size, entity_size, target, mutation_prob, population_size, max_iter
            )
            for _ in range(repetitions)
        ] for mutation_prob in mutation_probs
    ])

    # plot the results
    plot(results, mutation_probs, max_iter)


if __name__ == '__main__':
    main()
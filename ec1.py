import matplotlib.pyplot as plt

count = 0


def piechart(function_string, individuals, probs):
    global count
    count += 1

    plt.pie(probs, labels=individuals, autopct='%1.1f%%')
    # plt.title(function_string)
    plt.tight_layout()
    plt.savefig(f'results1/{count}.png')
    plt.clf()


def table(function_string, individuals, probs):
    print(function_string)
    for (x, prob) in zip(individuals, probs):
        print(f'{x} {prob:.4f}')
    print()


def sample_probabilities(fitness_function, individuals):
    fitnesses = list(map(fitness_function, individuals))
    summed_fitness = sum(fitnesses)
    return list(map(lambda x: x / summed_fitness, fitnesses))


def main():
    fitness_functions = [
        (lambda x: abs(x),     'f_1(x) = |x|'),
        (lambda x: x * x,      'f_2(x) = x^2'),
        (lambda x: 2 * x * x,  'f_3(x) = 2x^2'),
        (lambda x: x * x + 20, 'f_4(x) = x^2 + 20'),
    ]
    individuals = [2, 3, 4]

    for (fitness_function, function_string) in fitness_functions:
        probs = sample_probabilities(fitness_function, individuals)
        table(function_string, individuals, probs)
        piechart(function_string, individuals, probs)


if __name__ == '__main__':
    main()
import random
import math
import numpy as np


def int_to_base(n, base=16, min_digits=4):
    digits = "0123456789abcdef"
    if n == 0:
        return "0" * min_digits
    result = ""
    while n > 0:
        result = digits[n % base] + result
        n = n // base
    if len(result) < min_digits:
        result = "0" * (min_digits - len(result)) + result
    return result


def base_to_int(s, base=16):
    digits = "0123456789abcdef"
    s = s.lower()
    result = 0
    for char in s:
        result = result * base + digits.index(char)
    return result


def create_population(pop_size, dim, bounds, encoding='hex', gene_len=4):
    population = []
    for _ in range(pop_size):
        chromosome = ""
        for _ in range(dim):
            value = random.uniform(bounds[0], bounds[1])
            normalized = (value - bounds[0]) / (bounds[1] - bounds[0])
            max_val = (16 ** gene_len - 1) if encoding == 'hex' else (2 ** (gene_len * 4) - 1)
            int_val = int(normalized * max_val)
            if encoding == 'hex':
                gene = int_to_base(int_val, 16, gene_len)
            else:
                gene = int_to_base(int_val, 2, gene_len * 4)
            chromosome += gene
        population.append(chromosome)
    return population


def decode_chromosome(chromosome, dim, bounds, encoding='hex', gene_len=4):
    gene_len_actual = gene_len if encoding == 'hex' else gene_len * 4
    values = []
    for i in range(dim):
        gene = chromosome[i * gene_len_actual:(i + 1) * gene_len_actual]
        base = 16 if encoding == 'hex' else 2
        int_val = base_to_int(gene, base)
        max_val = (16 ** gene_len - 1) if encoding == 'hex' else (2 ** (gene_len * 4) - 1)
        normalized = int_val / max_val
        coord = bounds[0] + normalized * (bounds[1] - bounds[0])
        values.append(coord)
    return values


def sphere_function(x):
    return sum(xi ** 2 for xi in x)


def rastrigin_function(x):
    n = len(x)
    return 10 * n + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def rosenbrock_function(x):
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))


def ackley_function(x):
    n = len(x)
    sum1 = sum(xi ** 2 for xi in x)
    sum2 = sum(math.cos(2 * math.pi * xi) for xi in x)
    return -20 * math.exp(-0.2 * math.sqrt(sum1 / n)) - math.exp(sum2 / n) + 20 + math.e


functions = {
    'sphere': sphere_function,
    'rastrigin': rastrigin_function,
    'rosenbrock': rosenbrock_function,
    'ackley': ackley_function
}


def evaluate_population(population, dim, bounds, func_name, encoding='hex', gene_len=4):
    fitness = []
    func = functions[func_name]
    for chrom in population:
        values = decode_chromosome(chrom, dim, bounds, encoding, gene_len)
        value = func(values)
        fitness.append(1.0 / (1.0 + abs(value)))
    return fitness


def tournament_selection(population, fitness, tournament_size=3):
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = indices[0]
    for idx in indices[1:]:
        if fitness[idx] > fitness[best_idx]:
            best_idx = idx
    return population[best_idx]


def rank_selection(population, fitness):
    sorted_indices = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
    n = len(population)
    ranks = list(range(n, 0, -1))
    total = sum(ranks)
    probs = [rank / total for rank in ranks]
    selected_idx = np.random.choice(sorted_indices, p=probs)
    return population[selected_idx]


def single_point_crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() > crossover_rate or len(parent1) <= 1:
        return parent1, parent2
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def double_point_crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() > crossover_rate or len(parent1) <= 2:
        return parent1, parent2
    point1 = random.randint(1, len(parent1) - 2)
    point2 = random.randint(point1 + 1, len(parent1) - 1)
    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]
    return child1, child2


def mutate(chromosome, encoding='hex'):
    rand_num = random.randint(1, 10)
    if rand_num <= 2:
        pos = random.randint(0, len(chromosome) - 1)
        chrom_list = list(chromosome)
        if encoding == 'hex':
            hex_symbols = '0123456789abcdef'
            new_gene = random.choice(hex_symbols)
        else:
            new_gene = '1' if chrom_list[pos] == '0' else '0'
        chrom_list[pos] = new_gene
        return ''.join(chrom_list)
    return chromosome


def genetic_algorithm(
        pop_size=100,
        generations=100,
        dim=2,
        bounds=(-5.12, 5.12),
        func_name='sphere',
        selection_method='tournament',
        crossover_method='single',
        encoding='hex',
        gene_len=4
):
    population = create_population(pop_size, dim, bounds, encoding, gene_len)
    best_chromosome = None
    best_fitness = -float('inf')

    for gen in range(generations):
        fitness = evaluate_population(population, dim, bounds, func_name, encoding, gene_len)

        current_best_idx = np.argmax(fitness)
        if fitness[current_best_idx] > best_fitness:
            best_fitness = fitness[current_best_idx]
            best_chromosome = population[current_best_idx]

        if gen % 20 == 0 or gen == generations - 1:
            best_values = decode_chromosome(best_chromosome, dim, bounds, encoding, gene_len)
            func_value = functions[func_name](best_values)
            print(f"Поколение {gen}: f={func_value:.6f}, решение={[f'{x:.4f}' for x in best_values]}")

        new_population = []
        new_population.append(best_chromosome)

        while len(new_population) < pop_size:
            if selection_method == 'tournament':
                p1 = tournament_selection(population, fitness)
                p2 = tournament_selection(population, fitness)
            else:
                p1 = rank_selection(population, fitness)
                p2 = rank_selection(population, fitness)

            if crossover_method == 'single':
                c1, c2 = single_point_crossover(p1, p2)
            else:
                c1, c2 = double_point_crossover(p1, p2)

            c1 = mutate(c1, encoding)
            c2 = mutate(c2, encoding)

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population[:pop_size]

    best_values = decode_chromosome(best_chromosome, dim, bounds, encoding, gene_len)
    return best_values, functions[func_name](best_values), best_chromosome


print("=" * 60)
print("ГЕНЕТИЧЕСКИЙ АЛГОРИТМ")
print("=" * 60)

print("\nТест 1: Sphere, турнир, одноточечный, hex")
best, value, chrom = genetic_algorithm(
    pop_size=300,
    generations=500,
    func_name='sphere',
    selection_method='tournament',
    crossover_method='single',
    encoding='hex'
)
print(f"Лучшее: {best}, f={value:.10f}")

print("\nТест 2: Rastrigin, ранг, двухточечный, hex")
best, value, chrom = genetic_algorithm(
    pop_size=300,
    generations=500,
    func_name='rastrigin',
    selection_method='rank',
    crossover_method='double',
    encoding='hex'
)
print(f"Лучшее: {best}, f={value:.10f}")

print("\nТест 3: Rosenbrock, турнир, одноточечный, bin")
best, value, chrom = genetic_algorithm(
    pop_size=300,
    generations=500,
    bounds=(-2.048, 2.048),
    func_name='rosenbrock',
    selection_method='tournament',
    crossover_method='single',
    encoding='bin'
)
print(f"Лучшее: {best}, f={value:.10f}")

print("\nТест 4: Ackley, ранг, двухточечный, hex")
best, value, chrom = genetic_algorithm(
    pop_size=300,
    generations=500,
    bounds=(-32.768, 32.768),
    func_name='ackley',
    selection_method='rank',
    crossover_method='double',
    encoding='hex'
)
print(f"Лучшее: {best}, f={value:.10f}")

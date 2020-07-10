import streamlit as st
import math
import numpy as np
import pandas as pd
import random
import hvplot
import hvplot.pandas
from holoviews import opts
import holoviews as hv
import panel as pn
hv.extension('bokeh', logo=False)

@st.cache
def create_population(population_size, vector_length):
    return np.random.rand(population_size, vector_length)

"Say we have an optimisation problem we want to solve... like, what's the perfect combination of blah and bleh,"

def problem(soln):
    return soln[0]**2 + soln[1]**2 - 1

def optimum():
    return [-1/math.sqrt(2), 1/math.sqrt(2)]
print(-1/math.sqrt(2), 1/math.sqrt(2))
"First, create a population"
population_size = 100
vector_length = 2
current_population = create_population(population_size, vector_length)

def plot_population(current_population):
    population_plot = hv.Points(current_population)
    population_plot.opts(color='b', marker='^', size=5)
    return population_plot

target_data = [optimum()]

def add_point(target_data, label):
    target = hv.Points(target_data)
    target.opts(color='r')
    label = hv.Labels({('x', 'y'): target_data, 'text': label},  ['x', 'y'], 'text')
    label.opts(
        opts.Labels(text_font_size='10pt', yoffset=0.05, xoffset=0.1, text_color='r'),
    )
    target.redim(x=hv.Dimension('x', soft_range=(-2, 5)))
    return target*label

# label.opts()
# target*label
# population_plot = pd.DataFrame(data=current_population).hvplot(kind='scatter')
st.write(hv.render(plot_population(current_population)*add_point(target_data, 'optimum')))

"Then, we need some way to check which of the population is the fittest. For the sake of the this problem, we're going to assume the _lower_ the result when we put our potential solution (a member of the population) through the problem, the better."
"Give our problem a solution... lower is better"
def assess_fitness(individual, problem):
    "Determines the fitness of an individual using the given problem"
    return problem(individual)

"Then we need a way to check, who's our fittest member of our community"
def find_current_best(population, problem):
    """Evaluates a given population and returns the fittest individual"""
    fitnesses = [assess_fitness(x, problem) for x in population]
    best_value = min(fitnesses) # Lowest is best
    best_index = fitnesses.index(best_value)
    return best_index, fitnesses

"Now, we're going to let these potential solutions fight it out and only let a certain few have offspring."

best_index, fitnesses = find_current_best(current_population, problem)

def plot_best_and_target(current_population):
    return hv.render(plot_population(current_population)*add_point(target_data, 'optimum')*add_point([current_population[best_index]], 'current_best'))

st.write(plot_best_and_target(current_population))

def tournament_select_with_replacement(population, tournament_size, problem):
    "Competes a number of challengers and returns the fittest one"
    challengers_indexes = np.random.choice(population.shape[0], tournament_size, replace=True)
    challengers = population[challengers_indexes]
    best_index, _ = find_current_best(challengers, problem)
    return challengers[best_index]

def crossover(parent_a, parent_b):
    "Performs two point crossover on two parents"
    l = parent_a.shape[0]
    c, d = random.randint(0, l), random.randint(0, l)
    
    # Flip if c greater than d
    if (c > d): d, c = c, d 
    if (c == d): d += 1
    temp = np.copy(parent_a)
    child_a = np.concatenate([parent_a[0:c], parent_b[c:d], parent_a[d:]])
    child_b = np.concatenate([parent_b[0:c], temp[c:d], parent_b[d:]]) 
    return child_a, child_b

def mutate(child, mutation_rate, mutation_scale):
    "May mutate a child using Gaussian convolution"
    if mutation_rate >= random.uniform(0, 1):
        size = child.shape[0]
        mutation_value = np.random.normal(0, mutation_scale, size)
        child = child + mutation_value
    return child

def evolve(current_population, num_iterations=200):
    """Evolves the population for a given number of iterations

    Parameters
    ----------
    num_iterations : int, optional
        The number of generations before stopping evolving
    
    Returns
    -------
    results : list
        A list of the best fitness at each generation
    found_global_min_at : int
        The position at which the global min was found at, return -1 if not found
    """
    results = []
    found_global_min_at = -1
    best_possible_fitness = 0
    for i in range(num_iterations):
        best_index, fitnesses = find_current_best(current_population)
        results.append(fitnesses[best_index])
        if math.isclose(best_possible_fitness, fitnesses[best_index], rel_tol=1e-1):
            if found_global_min_at == -1:
                found_global_min_at = i
        current_population = update_population(current_population)
    return results, found_global_min_at

def update_population(current_population, problem):
    """Performs one generational update of Genetic Algorithm"""
    pop_size = len(current_population)
    next_population = np.empty((pop_size, 2))
    tournament_size = 2
    mutation_rate, mutation_scale = 0.3, 12
    for i in range(int(pop_size / 2)):
        parent_a = tournament_select_with_replacement(current_population, tournament_size, problem)
        parent_b = tournament_select_with_replacement(current_population, tournament_size, problem)
        child_a, child_b = crossover(parent_a, parent_b)
        next_population[i] = mutate(child_a, mutation_rate, mutation_scale)
        position_child_b = i + (pop_size / 2)
        next_population[int(position_child_b)] = mutate(child_b, mutation_rate, mutation_scale)
        # next_population = np.clip(next_population, self.problem.min_bounds, self.problem.max_bounds)
    return next_population

st.bokeh_chart(plot_best_and_target(current_population))

def update_plot(current_population):
    current_population = update_population(current_population, problem)
    st.write(plot_best_and_target(current_population))
    return current_population

# for x in range(10):
if st.button('Update Population'):
    current_population = update_plot(current_population)


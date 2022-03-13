import random

class GeneticAlgorithm:

    def _generate_genome(self, n_genes):
        """
        Function generates genome based on the number of genes.
        """
        return [random.randint(0, 1) for _ in range(n_genes)]

    def _generate_population(self, n_pops, n_genes):
        """
        Function generates n_pops sized population with random genome.
        """
        return [self._generate_genome(n_genes) for _ in range(n_pops)]

    def _selection(self, population, scores):
        """
        Function selects list parents.
        """
        return random.choices(population, weights=scores, k=int(len(population)/2))

    def _crossover(self, p1, p2):
        """
        Function picks crossover point and creates children.
        """
        crossover_point=random.randint(0, len(p1)-1)

        child1=p1[:crossover_point]+p2[crossover_point:]
        child2=p2[:crossover_point]+p1[crossover_point:]

        return child1, child2
    
    def _mutation(self, pop, mutation_chance):
        """
        Function performs mutation.
        """
        if random.random() < mutation_chance:
            mutation_point=random.randint(0, len(pop)-1)
            pop[mutation_point]=1-pop[mutation_point]

        return pop
    
    def run_evolution(self, fitness_function, fitness_limit, n_iterations, n_pops=100, n_genes=8, mutation_chance=0.05):
        """
        Main function which performs evolution.

        Arguments:
        fitness_function (function): function which will evaluate our population
        fitness_limit [double]: limit at which we stop performing evolution
        n_iterations [int]: number of generations
        n_pops [int]: number of pops in population
        n_genes [int]: number of genes each pop has
        mutation_chance [double]: probability of mutation
        """
        # generate starting population
        population=self._generate_population(n_pops, n_genes)

        # setting best pop and its evaluated value as first pop
        best_pop, best_eval = population[0], fitness_function(population[0])

        for i in range(n_iterations):

            print("Generation {}: best pop={}, value={}".format(i, best_pop, best_eval))
            # break early if condition satisfied
            if best_eval >= fitness_limit:
                break

            # evaluate
            scores=[fitness_function(pop) for pop in population]

            # find new best solution
            for j in range(len(population)):
                if scores[j] > best_eval:
                    best_pop, best_eval=population[j], scores[j]

            # selection
            parent1=self._selection(population, scores)
            parent2=self._selection(population, scores)

            children=[]
            for p1, p2 in zip(parent1, parent2):

                # crossover
                child1, child2 = self._crossover(p1, p2)

                # mutation
                child1=self._mutation(child1, mutation_chance)
                child2=self._mutation(child2, mutation_chance)

                # adding to child population
                children.append(child1)
                children.append(child2)
            
            population=children
        
        return best_pop, best_eval
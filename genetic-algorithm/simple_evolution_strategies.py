import numpy as np
import random

class EvolutionStrategy:
    def _generate_genome(self, n_genes):
        """
        Function generates genome based on the number of genes.
        """
        return np.random.random(n_genes)

    def _generate_population(self, n_pops, n_genes):
        """
        Function generates n_pops sized population with random genome.
        """
        return np.array([self._generate_genome(n_genes) for _ in range(n_pops)])

    def _generate_mutation_parameters(self, n_genes):
        """
        Function generates mutation parameters for each gene.
        """
        return np.random.random(n_genes)

    
    def _selection(self, population, max_parents):
        """
        Function selects list parents.
        """
        idx=np.random.choice(len(population), size=random.randint(2, max_parents), replace=True)
        return population[idx]

    def _crossover(self, parents):
        """
        Function performs crossover by counting mean in each gene column.
        """
        return np.mean(parents, axis=0)

    def _mutation(self, pop, mutation_chance):
        if random.random() < mutation_chance:
            # mutate mutation parameters by multiplying vector drawn from normal distribution
            self.mutation_parameters=self.mutation_parameters*np.random.randn(len(self.mutation_parameters))

            # mutate pop
            pop=pop+self.mutation_parameters

        return pop


    
    def run_evolution(self, fitness_function, fitness_limit, n_iterations, n_pops=100, n_genes=8, max_parents=4, mutation_chance=0.5):
        """
        Main function which performs evolution.

        Arguments:
        fitness_function (function): function which will evaluate our population
        fitness_limit [double]: limit at which we stop performing evolution
        n_iterations [int]: number of generations
        n_pops [int]: number of pops in population
        n_genes [int]: number of genes each pop has
        max_parents [int]: max number of parents  child can have
        mutation_chance [double]: probability of mutation
        """
        # generate starting population
        population=self._generate_population(n_pops, n_genes)

        # generate vector of mutation parameters
        self.mutation_parameters=self._generate_mutation_parameters(n_genes)

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

            # selection, multiple parents
            parent_list=np.array([self._selection(population, max_parents) for _ in range(len(population))], dtype=object)

            children=[]

            for parents in parent_list:
                # crossover
                child=self._crossover(parents)

                # mutation
                child=self._mutation(child, mutation_chance)

                # adding child to children list
                children.append(child)

            # concatenating parents and children
            parents_and_children=np.concatenate([population, children])

            # calculate fitness for each pop
            scores=np.array([fitness_function(pop) for pop in parents_and_children])

            # sorting scores and getting their indices
            sort_scores=np.argsort(scores)

            # selecting best pops from both children and parents
            population=parents_and_children[sort_scores[::-1]][:n_pops]
        
        return best_pop, best_eval

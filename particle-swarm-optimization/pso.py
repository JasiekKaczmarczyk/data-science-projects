import numpy as np

class ParticleSwarm:

    def __init__(self, search_area=np.array([[-10, 10], [-10, 10]]), update_function="bare_bones",  **kwargs):
        """
        
        Arguments:
        search_area (array):
        """
        self.cognitive=kwargs.get('cognitive', 2.05)
        self.social=kwargs.get('social', 2.05)
        self.inertia=kwargs.get('inertia', 0.7298)
        self.search_area=search_area


        self.update_functions= {
            "bare_bones": self._bare_bones
        }[update_function]

    def _generate_pop(self, n_dims):
        return (self.search_area[:, 1]-self.search_area[:, 0])*np.random.random(n_dims)+self.search_area[:, 0]

    def _generate_population(self, n_pops, n_dims):
        return np.array([self._generate_pop(n_dims) for _ in range(n_pops)])

    def _bare_bones(self, pop, best_pop):

        mu=(pop+best_pop)/2
        sigma=np.abs(pop-best_pop)

        return np.random.normal(mu, sigma)

    def _canonical_variant(self, pop, best_pop, previous_pop, v):
        indivialist_tendency=np.random.uniform(0, self.cognitive)*(previous_pop-pop)
        social_tendency=np.random.uniform(0, self.social)*(best_pop-pop)

        v=self.inertia*(v+indivialist_tendency+social_tendency)

        return pop+v

    def optimize(self, function_to_optimize, n_iterations=300, n_pops=16, n_dims=2):
        # initialize history
        self.history=np.zeros(shape=(n_iterations, n_dims))

        population=self._generate_population(n_pops, n_dims)

        best_pop, best_eval = population[0], function_to_optimize(population[0])

        for i in range(n_iterations):  
            # evaluate
            scores=np.apply_along_axis(function_to_optimize, 1, population)

            best_idx = np.argsort(scores)[0]
            best_pop, best_eval = population[best_idx], function_to_optimize(population[best_idx])

            # update history
            self.history[i] = best_pop

            print("Generation {}: best pop={}, value={}".format(i, best_pop, best_eval))

            # generate new coordinates
            new_population=np.apply_along_axis(self.update_functions, 1, population, best_pop=best_pop)
            new_scores=np.apply_along_axis(function_to_optimize, 1, new_population)

            # creating mask which evaluates if new position is better
            mask=new_scores<scores

            # moving pop if new score is better
            population[mask]=new_population[mask]
            
        return best_pop, best_eval

    def get_history(self):
        return self.history
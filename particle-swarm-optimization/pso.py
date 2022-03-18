import numpy as np

class ParticleSwarm:

    def __init__(self, search_area=np.array([[-10, 10], [-10, 10]]), update_function="bare_bones",  **kwargs):
        """
        
        Arguments:
        search_area (array): area on which initial population will be spawned
        update_function [string]: name of function which will be updating pops

        **kwargs {"cognitive": <val>, "social": <val>, "inertia": <val>}: parameters which specify behaviour of pops
        """
        self.cognitive=kwargs.get('cognitive', 2.05)
        self.social=kwargs.get('social', 2.05)
        self.inertia=kwargs.get('inertia', 0.7298)
        self.search_area=search_area


        self.update_function_name=update_function

        self.update_function={
            "bare_bones": self._bare_bones,
            "canonical": self._canonical_variant
        }[update_function]

    def _generate_pop(self, n_dims, search_area):
        return (search_area[:, 1]-search_area[:, 0])*np.random.random(n_dims)+search_area[:, 0]

    def _generate_population(self, n_pops, n_dims, search_area):
        return np.array([self._generate_pop(n_dims, search_area) for _ in range(n_pops)])

    def _bare_bones(self, population, best_pop, previous_population, v):
        """
        Bare bones update function
        
        previous_population and v are not used but are provided so that all update functions have the same set od arguments
        """
        mu=(population+best_pop)/2
        sigma=np.abs(population-best_pop)

        return np.random.normal(mu, sigma)

    def _canonical_variant(self, population, best_pop, previous_population, v):
        """
        Canonical variant of update function
        """
        indivialist_tendency=np.random.uniform(0, self.cognitive)*(previous_population-population)
        social_tendency=np.random.uniform(0, self.social)*(best_pop-population)

        v=self.inertia*(v+indivialist_tendency+social_tendency)

        return population+v


    def optimize(self, function_to_optimize, n_iterations=300, n_pops=16, n_dims=2):
        """
        Main method which optimizes given function
        """

        # initialize history
        self.history=np.zeros(shape=(n_iterations, n_dims))

        # generate population
        population=self._generate_population(n_pops, n_dims, self.search_area)

        # initializing best pop and it's value
        best_pop, best_eval = population[0], function_to_optimize(population[0])

        # initializing previous population (on smaller search area) and v
        previous_population=self._generate_population(n_pops, n_dims, self.search_area/2)
        v=np.zeros(shape=(n_pops, n_dims))

        for i in range(n_iterations):
            # if i > 30:
            #     if np.mean(np.std(np.apply_along_axis(function_to_optimize, 1, self.history[i-30:i]), axis=0))  < 0.001:
            #         break

            # evaluate
            scores=np.apply_along_axis(function_to_optimize, 1, population)

            best_idx = np.argsort(scores)[0]
            best_pop, best_eval = population[best_idx], function_to_optimize(population[best_idx])

            # update history
            self.history[i] = best_pop

            print("Generation {}: best pop={}, value={}".format(i, best_pop, best_eval))

            # generate new coordinates using specified update function
            new_population=self.update_function(population, best_pop, previous_population, v)

            # updating previous_population and v if update method is not bare bones to prevent unnecessary calculations
            if self.update_function_name != "bare_bones":
                v=new_population-population
                previous_population=population

            # evaluating new population
            new_scores=np.apply_along_axis(function_to_optimize, 1, new_population)

            # creating mask which evaluates if new position is better
            mask=new_scores<scores

            # moving pop if new score is better
            population[mask]=new_population[mask]

 
            
        return best_pop, best_eval

    def get_history(self):
        return self.history
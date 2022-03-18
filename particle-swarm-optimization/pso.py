import numpy as np

class ParticleSwarm:

    def __init__(self, spawn_area=np.array([[-10, 10], [-10, 10]]), update_function="bare_bones",  **kwargs):
        """
        Particle Swarm Optimization Algorithm

        Arguments:
        spawn_area (array): area on which initial population will be spawned
        update_function [string]: name of function which will be updating pops

        **kwargs {"cognitive": <val>, "social": <val>, "inertia": <val>}: parameters which specify behaviour of pops
        """
        self.cognitive=kwargs.get('cognitive', 2.05)
        self.social=kwargs.get('social', 2.05)
        self.inertia=kwargs.get('inertia', 0.7298)
        self.spawn_area=spawn_area

        # this variable will only be used to determine if we need to calculate some other parameters for certain function
        self.update_function_name=update_function

        # choosing update function from dict of available update functions
        self.update_function={
            "bare_bones": self._bare_bones,
            "canonical": self._canonical_variant
        }[update_function]

        # nr of last iteration
        self.end_iteration=-1

    def _generate_pop(self, n_dims, spawn_area):
        """
        Initializes coordinates for single pop in certain search area
        """
        return (spawn_area[:, 1]-spawn_area[:, 0])*np.random.random(n_dims)+spawn_area[:, 0]

    def _generate_population(self, n_pops, n_dims, spawn_area):
        """
        Generates population
        """
        return np.array([self._generate_pop(n_dims, spawn_area) for _ in range(n_pops)])

    def optimize(self, function_to_optimize, n_iterations=300, n_pops=16, n_dims=2):
        """
        Main method which optimizes given function

        Arguments:
        function_to_optimize (function): function we want to find minimum
        n_iterations [int]: number of iterations of algorithm
        n_pops [int]: size of population
        n_dims [int]: number of dimensions of given function
        """

        # initialize history of best pops which will be updated in each iteration
        self.history=np.zeros(shape=(n_iterations, n_dims))

        # generate population
        population=self._generate_population(n_pops, n_dims, self.spawn_area)

        # initializing best pop and it's value
        best_pop = population[0]
        best_eval = function_to_optimize(best_pop)

        # initializing previous population (on smaller search area) and v
        previous_population=self._generate_population(n_pops, n_dims, self.spawn_area/2)
        v=np.zeros(shape=(n_pops, n_dims))

        for i in range(n_iterations):

            # evaluate population
            scores=np.apply_along_axis(function_to_optimize, 1, population)

            # find best pop
            best_idx = np.argsort(scores)[0]
            best_pop = population[best_idx]
            best_eval = function_to_optimize(best_pop)

            # update history
            self.history[i] = best_pop

            # printing progress
            print("Generation {}: best pop={}, value={}".format(i, best_pop, best_eval))

            # generate new coordinates using specified update function (bare_bones, canonical, ...)
            new_population=self.update_function(population, best_pop, previous_population, v)

            # checking if not bare bones to prevent unnecessary calculations for this method
            if self.update_function_name != "bare_bones":
                # updating previous_population and v if update method
                v=new_population-population
                previous_population=population

            # evaluating new population
            new_scores=np.apply_along_axis(function_to_optimize, 1, new_population)

            # creating mask which evaluates if new position is better
            mask=new_scores<scores

            # moving pop if new score is better
            population[mask]=new_population[mask]

            # stop criterion if std of whole population is less than 0.001 (whole population converged in one spot)
            if np.std(new_scores) < 0.001:
                # setting up nr of last iteration
                self.end_iteration=i
                break
 
            
        return best_pop, best_eval

    def get_history(self):
        """
        Function returns history of how best pop coordinates changed
        """
        return self.history[:self.end_iteration]


    # AVAILABLE UPDATE FUNCTIONS
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
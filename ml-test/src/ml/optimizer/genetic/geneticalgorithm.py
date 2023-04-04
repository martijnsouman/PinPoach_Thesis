from .generation import *
import random
import copy

## GeneticAlgorithm class
#
# Used to search for the individual with the highest fitness value
class GeneticAlgorithm:
    _seed = None
    
    ## Constructor for the GeneticAlgorithm class
    # @param output_directory   The directory to write the results/checkpoints to
    # @param max_generations    The maximum amount of generations to evaluate
    # @param population_size    The size of one population for the genetic algorithm
    # @param breed_best         The number of individuals in the population which are considered as best are bred
    # @param breed_random       The number of individuals in the population are randomly bred with the best
    # @param mutation_deviation     Defines the mutation deviation for each parameter
    # @param fixed_seed The seed for the random generator
    def __init__(self,
        output_directory,
        max_generations,
        population_size, 
        breed_best,
        breed_random,
        mutation_deviation,
        fixed_seed=1):

        self._outputDirectory = output_directory
        self._maxGenerations = max_generations
        self._populationSize = population_size
        self._breedBestN = int(breed_best)
        self._breedRandomN = int(breed_random)
        self._mutationDeviation = mutation_deviation

        # Set fixed seed for deterministic results
        self._seed = fixed_seed
        random.seed(fixed_seed)

        # Create a log
        self._log = list()

    ## Search for the individual with the highest fitness
    # @param base_individual             The individual where the first population is derived from 
    # @param random_first_generation    Use random parameters for the first generation
    # @return                           The best individual (highest fitness value) after n generations
    def search(self, base_individual, random_first_generation=False):
        base_individual.setSeed(self._seed)

        # Create a base population
        basePopulation = [base_individual]

        # Set the temporarely best individual 
        bestIndividual = base_individual
        bestIndividual.setId(-1)
    
        # Run for n generations
        for n in range(0, self._maxGenerations):
            spawnRandom = (random_first_generation and n == 0)

            gen = Generation(n, self._outputDirectory, self._mutationDeviation, self._seed + n)
            gen.spawn(basePopulation, self._populationSize, spawnRandom)
            
            # Evaluate all individuals
            gen.evaluate()

            # Store checkpoint
            gen.storeCheckpoint()
            
            # Get the individuals
            best = gen.getBestIndividuals(self._breedBestN)
            rand = gen.getRandomIndividuals(self._breedRandomN)

            # Check if a better individual is found
            if (bestIndividual.getFitness() == None) or best[0].getFitness() > bestIndividual.getFitness():
                print("A new best individual is found with fitness {fitness}".format(fitness=best[0].getFitness()))

                # Set the new best individual
                bestIndividual = copy.copy(best[0])

            # Print info
            print("Generation {number} of {gens}:\n\tAverage fitness: {average}\n\tBest fitness: {best}\n\tAll time best fitness: {atb}".format(
                number=n+1, 
                gens=self._maxGenerations,
                average=gen.getAverageFitness(), 
                best=gen.getBestFitness(),
                atb=bestIndividual.getFitness())
            )
           
            # Generate offspring
            basePopulation = list()

            # Breed the best with best and random individuals
            basePopulation.extend(self._breedGroups(best, best))
            basePopulation.extend(self._breedGroups(best, rand))

        # Get the final results
        return bestIndividual 
    

    ## Breed two groups of individuals
    # @param a      Group a
    # @param b      Group b
    # @return       The resulting list of individuals
    def _breedGroups(self, a, b):
        return [x.breed(random.choice(b)) for x in a]

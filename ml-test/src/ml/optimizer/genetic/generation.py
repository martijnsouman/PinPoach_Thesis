import random
import json
import os

## Generation class
#
# Used to define a generation within the GeneticAlgorithm
class Generation:
    _individuals = None
    _seed = None
    
    ## Constructor for the Generation class
    # @param generation_n           The generation number
    # @param output_directory       The output directory for checkpoints, etc.
    # @param mutation_deviation     The mutation deviation 
    # @param fixed_seed             Seed for the random generator
    def __init__(self, generation_n, output_directory, mutation_deviation, fixed_seed):
        self._generationNumber = generation_n
        self._outputDirectory = output_directory
        self._individuals = list()
        self._mutationDeviation = mutation_deviation

        # Set fixed seed for deterministic results
        self._seed = fixed_seed
        random.seed(fixed_seed)

        # Create directory
        try:
            path = os.path.join(self._outputDirectory, "generation_{n}".format(n=generation_n))
            os.mkdir(path)
        except:
            pass

    
    ## Spawn a new population in this generation
    # @param base_population    The population to derive the new (larger) population from
    # @param population_size    The size of the new population to spawn
    # @param random_parameters  Randomize the parameters for every spawned individual
    def spawn(self, base_population, population_size, random_parameters=False):
        baseSpawnFactor = int(population_size/len(base_population))
        
        # Spawn the individuals
        for individual in base_population:
            for n in range(0, baseSpawnFactor):
                # Copy the individual
                newIndividual = individual.copy()

                # Check if the parameters should be random, or based on the base population
                if random_parameters:
                    # Use random parameters
                    newIndividual.randomizeParameters()
                else:
                    # Use mutated parameters
                    newIndividual.mutate(self._mutationDeviation)
                
                # Append the individual to this population
                self._individuals.append(newIndividual)


        # Assign an id and seed to each individual
        for n in range(0, population_size):
            self._individuals[n].setSeed((self._seed * population_size) + n)
            self._individuals[n].setId(n)

        print("Spawned {x} individuals.".format(x=len(self._individuals)))
    

    ## Evaluate this generation 
    #
    # Sorts all individuals based on fitness
    def evaluate(self):
        # Examine all individuals
        for i, individual in enumerate(self._individuals):
            print("\nIndividual {i} of {n}:".format(i=i+1, n=len(self._individuals)))
            individual.evaluate()

            # Store the individual data
            path = os.path.join(self._outputDirectory, "generation_{n}".format(n=self._generationNumber))
            individual.store(path)
        
        # Sort individuals based on fitness
        self._individuals.sort(key=lambda x: x.getFitness(), reverse=True)
        
        print("This generation fitness values:")
        for individual in self._individuals:
            print("Fitness: {f}".format(f=individual.getFitness()))
    

    ## Get the best n individuals from this generation
    # @param n  The amount of individuals
    # @return List of n individuals
    def getBestIndividuals(self, n):
        return self._individuals[:n]

    ## Get n random selected individuals from this generation
    # @param n  The amount of individuals
    # @return List of n individuals
    def getRandomIndividuals(self, n):
        return [random.choice(self._individuals) for i in range(0, n)]
    
    ## Get the average fitness value for this generation
    # @return Average fitness value
    def getAverageFitness(self):
        total = 0
        for i in self._individuals:
            total += i.getFitness()
        
        return total/len(self._individuals)
    

    ## Get the best fitness value for this generation
    # @return Best fitness value
    def getBestFitness(self):
        return self._individuals[0].getFitness()

    ## Get the worst fitness value for this generation
    # @return Worst fitness value
    def getWorstFitness(self):
        return self._individuals[-1].getFitness()

    ## Store a checkpoint for this generation
    def storeCheckpoint(self):
        #Add information about the best individual
        bestIndividual = self.getBestIndividuals(1)[0]
        data = {
            "best individual": bestIndividual.getId()
        }

        # Write to a file
        path = os.path.join(
            self._outputDirectory,
            "generation_{n}".format(n=self._generationNumber),
            "results.json"
        )
        with open(path, "w+") as jsonFile:
            json.dump(data, jsonFile)

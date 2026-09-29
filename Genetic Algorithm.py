import random

# -----------------------------------------
# HOME RENT APPLICATION USING GENETIC ALGORITHM
# -----------------------------------------

# Available rental houses
# Format: (House ID, [BHK, Size in sqft, Rent])
houses = [
    (1, [2, 850, 1100]),
    (2, [3, 1150, 1450]),
    (3, [4, 1600, 2200]),
    (4, [3, 1250, 1550]),
    (5, [1, 500, 700]),
    (6, [2, 1000, 1300]),
    (7, [3, 1300, 1500]),
    (8, [4, 1800, 2500]),
    (9, [2, 950, 1200]),
    (10, [3, 1200, 1400])
]


# -----------------------------------------
# GET USER REQUIREMENTS
# -----------------------------------------

print("======================================")
print("       HOME RENT APPLICATION")
print("======================================")

target_bhk = int(input("Enter preferred number of BHK: "))
target_size = int(input("Enter preferred house size (sqft): "))
target_rent = int(input("Enter maximum monthly rent: "))

# User's requirements become the TARGET
TARGET = [target_bhk, target_size, target_rent]


# -----------------------------------------
# FITNESS FUNCTION
# -----------------------------------------

def fitness(house):

    bhk = house[0]
    size = house[1]
    rent = house[2]

    # Difference from user's requirements
    bhk_diff = abs(bhk - TARGET[0]) * 200

    size_diff = abs(size - TARGET[1])

    # Heavy penalty if rent exceeds user's budget
    rent_diff = max(0, rent - TARGET[2]) * 2

    # Lower fitness = better house
    return bhk_diff + size_diff + rent_diff


# -----------------------------------------
# GENETIC ALGORITHM
# -----------------------------------------

def run_ga():

    pop_size = 4
    generations = 50

    # Randomly initialize population
    population = random.choices(houses, k=pop_size)

    for generation in range(generations):

        # Sort according to fitness
        population = sorted(
            population,
            key=lambda x: fitness(x[1])
        )

        # Best house
        best_house = population[0]

        # Stopping condition
        if fitness(best_house[1]) < 50:
            break

        # Elitism: keep best solution
        next_generation = [best_house]

        # Generate remaining population
        while len(next_generation) < pop_size:

            # Selection: choose one of the best houses
            parent = random.choice(population[:2])

            # Mutation
            if random.random() < 0.3:
                child = random.choice(houses)
            else:
                child = parent

            next_generation.append(child)

        population = next_generation

    return population[0]


# -----------------------------------------
# RUN GENETIC ALGORITHM
# -----------------------------------------

best_match = run_ga()


# -----------------------------------------
# DISPLAY RESULT
# -----------------------------------------

print("\n======================================")
print("       BEST HOUSE RECOMMENDATION")
print("======================================")

print("House ID      :", best_match[0])
print("BHK           :", best_match[1][0])
print("Size          :", best_match[1][1], "sqft")
print("Monthly Rent  : $", best_match[1][2])
print("Fitness Score :", fitness(best_match[1]))

print("\nLower fitness score = Better match")

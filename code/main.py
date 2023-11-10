import numpy as np

### Hyperparameters ###
S0 = 2000 # seeds dispersed per puff ball
ld_death_seeds = 4 # average life expectancy of seeds in days
ld_death_dandelion = 15 # average time to get consumed
germ_mean = 12 # average germination time in days
optimal_time_to_grow = 50 # optimal time to grow in days
sim_length = 365 # simulation length in days
min_dist = 1 # minimum distance between seeds in meters
side_length = 100 # side length of square plot in meters

### DEFAULT (TEMPERATE) REGIONAL DATA ###
wind_mean = [int(np.random.normal(loc=100, scale=100)) for i in range(365)] # average wind speed in mph
temperatures = [int(20*np.sin(2*np.pi* x/(360))+55) for x in range(365)]
light_hours = [int(6*np.sin(2*np.pi* x/(360))+10) for x in range(365)]
k = [int(np.random.randint(10, 100)) for x in range(365)]
n = [int(1.05*np.random.randint(0, 50)) for x in range(365)]

class Seed:
    ### CONSTRUCTOR ###
    def __init__(self, t, x, y):
        self.t = t
        self.growth_state = 0
        self.coordinate = [x, y]
        self.death_date = self.get_death_day(t, ld_death_seeds)
        self.germination_date = self.get_germination_date(t+1)
        self.germination_state = 0

    ### STAGE 1: BROWNIAN MOTION ###
    def brownian_motion(self):
        ### POISSON DISTRIBUTION FOR NUM STEPS ###
        def num_steps(t):
            m = [abs(wind_mean[t % 365]) for x in range(365)]
            num = int(np.random.poisson(m[t % 365], 1))
            return num

        ### BROWNIAN RANDOM WALK ###
        steps = num_steps(0)
        for i in range(steps):
            self.coordinate[0] += np.random.normal(loc=0, scale=2)
            self.coordinate[1] += np.random.normal(loc=0, scale=2)
        
    ### STAGE 2: CALCULATE DEATH DATE ###
    def get_death_day(self, t, mean):
        ld = [int(mean*np.sin(2*np.pi* x/(360))+(mean/2)) for x in range(365)]     
        #death_time = t+int(np.random.exponential(ld[t % 365]))
        death_time = t+int(np.random.exponential(mean))
        return death_time
    
    ### STAGE 3: SEED GERMINATION ###
    def get_germination_date(self, t):
        return t+int(np.random.normal(germ_mean, germ_mean))

    def dGdt(self, t):
        def T_I(temp, max=77, min=50):
            middle = (max+min) / 2
            if temp > max:
                return -2 / (1 + np.exp(-0.1*(temp - max))) + 1
            elif temp < min:
                return -2 / (1 + np.exp(-0.1*(min-temp))) + 1
            else:
                return 1 - (1/(middle-min)**2) * (temp-middle)**2
        
        def L_I(light_hrs, max=24, min=6):
            middle = (max+min) / 2
            if light_hrs > max:
                return -2 / (1 + np.exp(-0.5*(light_hrs - max))) + 1
            elif light_hrs < min:
                return -2 / (1 + np.exp(-0.5*(min-light_hrs))) + 1
            else:
                return 1 - (1/(middle-min)**2) * (light_hrs-middle)**2
        
        def N_I(k, n, k_max=80, k_min=40, n_max=40, n_min=20):
            k_avg = (k_max+k_min)/2
            n_avg = (n_max+n_min)/2
            k_score = np.exp(-((k-k_avg)/(k_max-k_min))**2)
            n_score = np.exp(-((n-n_avg)/(n_max-n_min))**2)
            return k_score + n_score - 1
        
        return (1/(3 * optimal_time_to_grow)) * (T_I(temperatures[t % 365]) + L_I(light_hours[t % 365]) + N_I(k[t % 365], n[t % 365]))

### INITIAL STATE ###
seeds = [Seed(0, 0, 0) for i in range(S0)]
for seed in seeds:
    seed.brownian_motion()
    if (seed.coordinate[0] < 0 or seed.coordinate[1] < 0 or seed.coordinate[0] > side_length or seed.coordinate[1] > side_length):
        seeds.remove(seed)

### FILTERING COORDINATES
def euclidean_distance(coord1, coord2):
    x1, y1 = coord1
    x2, y2 = coord2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def remove_close_seeds(seeds, threshold=min_dist):
    filtered_coordinates = []
    n = len(seeds)
    for i in range(n):
        keep = True
        for j in range(i + 1, n):
            if euclidean_distance(seeds[i].coordinate, seeds[j].coordinate) < threshold:
                keep = False
                break
        if keep:
            filtered_coordinates.append(seeds[i])
    return filtered_coordinates

seeds = remove_close_seeds(seeds)
population_seeds = []
population_plants = []
### BEGIN SIMULATION ###
for t in range(1, sim_length): 
    x = sum(1 for seed in seeds if seed.germination_state == 0)
    y = sum(1 for seed in seeds if seed.germination_state == 1)
    population_seeds.append(x)
    population_plants.append(y)
    
    for i in range(len(seeds)-1, -1, -1):
        seed = seeds[i]
        
        ### CHECK DANDELION DEATH ###
        if seed.death_date == t and seed.germination_state == 0: # before growth stage
            seeds.pop(i)
        elif seed.death_date == t and seed.germination_state == 1: # in growth stage
            seed.growth_state = 0
            seed.death_date = seed.get_death_day(t, ld_death_dandelion)
        seed.t = t
        
        ### CHECK GERMINATION ###
        if seed.germination_date == t:
            seed.germination_state = 1
            seed.death_date = seed.get_death_day(t+1, ld_death_dandelion)
        
        ### CHECK PLANT GROWTH ###
        if seed.germination_state == 1:
            seed.growth_state += seed.dGdt(t)
            
            ### REACHES PUFF BALL STAGE
            if (seed.growth_state < 0):
                seed.growth_state = 0
            
            if (seed.growth_state >= 1):
                seed.growth_state = 0
                seed.death_date = seed.get_death_day(t+1, ld_death_dandelion)
                
                ### BEGIN BROWNIAN MOTION AGAIN
                new_seeds = [Seed(t, seed.coordinate[0], seed.coordinate[1]) for i in range (S0)]
                for new_seed in new_seeds:
                    new_seed.brownian_motion()
                    if (new_seed.coordinate[0] < 0 or new_seed.coordinate[1]< 0 or new_seed.coordinate[0] > side_length or new_seed.coordinate[1] > side_length):
                        new_seeds.remove(new_seed)
                seeds += new_seeds
                seeds = remove_close_seeds(seeds)

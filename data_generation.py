import numpy as np
import pandas as pd

def autoscale(x):
    return (x - np.mean(x)) / np.std(x)

def rescale(x, a_range, b_range):
    a = np.random.uniform(a_range[0], a_range[1], size=1)
    b = np.random.uniform(b_range[0], b_range[1], size=1)
    return a + b * x

def generate(latent, n_obs, b_range=(0.2,2), noise_sd=1):
    observed = np.random.normal(scale=noise_sd, size=n_obs) # noise
    for i in range(latent.shape[1]):
        slope = np.random.uniform(b_range[0], b_range[1], size=1)
        observed += slope * latent[:,i]
    return autoscale(observed)

np.random.seed(777)

# Train data -------------------------------------------------------------------

n_train = 100

latent = np.random.normal(size=(n_train, 2))
train_data = np.zeros((n_train, 5))
variables = []
for j in range(5):
    train_data[:,j] = generate(latent, n_obs=n_train)
    variables.append("var"+str(j+1))

pd.DataFrame(train_data).corr()

# Test data --------------------------------------------------------------------

n_test = 30

n1 = int(np.ceil(n_test/2))
n2 = n_test - n1
apex = 16
trend1_increasing = np.linspace(start=0, stop=apex, num=n1)
trend2_stationary = np.repeat(apex, n2)
trend2_decreasing = np.linspace(start=apex, stop=0, num=n2)

typeA = np.sqrt(np.hstack([trend1_increasing,trend2_stationary]))
typeB = np.sqrt(np.hstack([trend1_increasing,trend2_decreasing]))

test_data = np.transpose(np.array([typeA, typeA, typeA, typeA, typeB]))

# Tabulate data ----------------------------------------------------------------

train_data = pd.DataFrame(train_data, columns=variables)
train_data.insert(loc=0, column="group", value="train")
train_data.insert(loc=1, column="time", value=[*range(1,n_train+1)])

test_data = pd.DataFrame(test_data, columns=variables)
test_data.insert(loc=0, column="group", value="test")
test_data.insert(loc=1, column="time", value=[*range(1,n_test+1)])

# Rescale data -----------------------------------------------------------------

scale_pars = {
    "var1": {"a": (87,92), "b": (8,12)},
    "var2": {"a": (29,31), "b": (4,6)},
    "var3": {"a": (97,1002), "b": (14,16)},
    "var4": {"a": (15,16), "b": (1.8,2.2)},
    "var5": {"a": (236,238), "b": (12,14)}
}

plant = pd.concat([train_data, test_data])

for j in variables:
    plant[j] = rescale(
        plant[j],
        a_range=scale_pars[j]["a"],
        b_range=scale_pars[j]["b"]
    )

plant.to_csv("plant_simulated.csv", index=False)

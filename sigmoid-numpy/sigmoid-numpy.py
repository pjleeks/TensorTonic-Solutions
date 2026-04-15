import numpy as np

def sigmoid(x):
    # Convert input to a numpy array to allow element-wise operations
    x = np.array(x) 
    return 1 / (1 + np.exp(-x))

# Now this will work:
my_list = [0, 2, -2]
print(sigmoid(my_list))

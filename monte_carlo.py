import os
import random
import numpy as np

random.seed(42)

path = "./input/test_v2/test"
 
obj = os.scandir(path=path)

print(str(obj))


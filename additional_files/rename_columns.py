# This script should take a .scv in the format: 
#           X,Y,Z,id,elevation_points1
#           155.494883601,-327.374549139,0,"0",230.926942429935
#           175.494883601,-327.374549139,0,"1",221.755966342484
#
# And output a .csv in the format: 
#           x,y,z
#           155.494883601,-327.374549139,230.926942429935
#           175.494883601,-327.374549139,221.755966342484
#   -Sjur Barndon

import os
import sys 
import numpy as np
import pandas as pd

# Sets python directory:
os.chdir(os.path.dirname(sys.argv[0]))
dir = os.path.dirname(os.path.realpath(__file__))

# Set path names:
path_input = "sampled_domain_20m.csv"
path_save = "sampled_domain_20m_clean.csv"

# Load SCV:
csv_pd = pd.read_csv(path_input)
print(csv_pd.head(2))                               #show first two rows

# Make new pandas list: 
new_pd = csv_pd[["X", "Y", "elevation_points1"]]    #get only xyz-data columns
new_pd.columns=["x", "y", "z"]                      #rename columns to z,y,z
print(new_pd.head(2))                               

# Save as new .csv-file:
new_pd.to_csv(path_save, index=False)
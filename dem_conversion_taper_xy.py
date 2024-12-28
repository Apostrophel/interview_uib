# DEM Conversion Taper
# This scrips converts a DEM in csv to npy and adds 'taper'-regions along both axes. 
# This is intended for converting GIS data into usable elevation data in Elmer/Ice. 
# The exported npy file must be converted to a .dat file with the matlab script npy_to_xyz.m
#
# Steps: 
#    1. Convert .csv data from a QGIS raster export to "xyz" formatted numpy array 
#              - Also convert from longitude, latitude to x, y starting from 0 (By normalizing the data).
#              - Convert to a numpy 'matrix-shaped' 2d-array with z-values as values for x-y indices.
#    2. Add a taper length at downflow end of domain that loops back to same elevation as x = 0
#    3. Apply a gaussian filter (smoothing out roads, houses and other general inaccuarcies)
#    4. Output a .npy file. This output will be used in a matlab script for .dat convertion.
#
# Sjur Barndon, sjurbarndon@proton.me

import os
import sys 
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

# Set python directory:
os.chdir(os.path.dirname(sys.argv[0]))
dir = os.path.dirname(os.path.realpath(__file__))

# Define input dimensions:
x_in_orig = 500             #(m) x input domain size 
y_in_orig = 250             #(m) y input domain size 

resolution = 1              #(m) resolution                            
g_sd = 1.5 #50 #1.5         #Gaussian standard deviation value. Use 50 for smoothed test DEM.

# Set load and save paths:
path_dem_domain = "sampled_domain_20m_clean.csv"     #250x500 = 125 000 points. This defines both y and x_in_orig above:
path_output = "veafjorden_topography_r20_8k_4k_shift.npy"

#Show or hide visual plot outputs during processing:
show_plots = True          

# Functions:
def get_middle_profile_vector(my_matrix):
    """
    Returns a profile of the centre along the x-axis, for comparative plotting.
    """
    return my_matrix[int(y_in_orig/2)].flatten()

def get_profile_vector_at(input_x, my_matrix):
    """
    Returns a profile at x coordinate for comparison plot. 
    """
    start_row = 0
    end_row = y_in_orig # This is exclusive
    start_col = input_x
    end_col = input_x+1
    middle_array = my_matrix[start_row:end_row, start_col:end_col]
    return middle_array.flatten()

def hillshade(array,azimuth,angle_altitude):
    """
    Generate a hillshade of the topography. (Written by Robert Law).
    """
    azimuth = 360.0 - azimuth 

    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azimuthrad = azimuth*np.pi/180.
    altituderad = angle_altitude*np.pi/180.

    shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)

    return 255*(shaded + 1)/2

def check_equal_boundaries(dem):
    """
    
    """

    n = 20  #number of output lines (for debugging)

    # Check if the left side equals the right side
    left_side = dem[:,:1]
    right_side = dem[:, -1:]
    left_right_equal = np.array_equal(left_side, right_side)

    # Check if the top equals the bottom
    top_edge = dem[0, :]
    bottom_edge = dem[-1, :]
    top_bottom_equal = np.array_equal(top_edge, bottom_edge)

    if left_right_equal:
        print("\nLeft and right sides of the DEM are equal.")
        #for l, r in zip(left_side[:n], right_side[:n]):
        #    print(l, " == ", r )
    else:
        print("\nLeft and right sides of the DEM are not exactly equal:")
        for l, r in zip(left_side[:n], right_side[:n]):
            print(l, " =/= ", r )

    if top_bottom_equal:
        print("\nTop and bottom of the DEM are equal.")
        #for t, b in zip(top_edge[:n], bottom_edge[:n]):
        #    print(t, " == ", b )
    else:
        print("\nTop and bottom of the DEM are not exactly equal:")
        #for t, b in zip(top_edge[:n], bottom_edge[:n]):
        #    print(t, " =/= ", b )
        for t, b in zip(top_edge, bottom_edge):
            if (t != b):
                print(t, " =/= ", b )
    
    return left_right_equal, top_bottom_equal
def check_equal_boundaries_visual(dem):
    col_1 = dem[:,:2]
    col_n = dem[:,-2:]
    plt.imshow(np.block([col_n, col_1]), aspect="auto")
    plt.title("check lateral boundaries")
    if show_plots: plt.show()
    
    row_1 = dem[0, :]
    row_n = dem[-1,:]
    plt.imshow(np.vstack((row_1, row_n)), aspect="auto")
    plt.title("Check Vertical Boundaries")
    if show_plots: plt.show()


#################################################################################
############                     Start of script                     ############       
#################################################################################


# Load csv data:
DEM_pd = pd.read_csv(path_dem_domain)

# Calculate relief: 
relief = abs(max(DEM_pd.z) - min(DEM_pd.z))
print("Max: ", max(DEM_pd.z))
print("min: ", min(DEM_pd.z))
print("Relief: ", relief)

# Normalise x, y (and z?):
DEM_pd['x'] = (DEM_pd['x']- DEM_pd['x'].min()) / (DEM_pd['x'].max() - DEM_pd['x'].min() )
DEM_pd['y'] = (DEM_pd['y']- DEM_pd['y'].min()) / (DEM_pd['y'].max() - DEM_pd['y'].min() )
DEM_pd['z'] = (DEM_pd['z']- DEM_pd['z'].min()) / (DEM_pd['z'].max() - DEM_pd['z'].min() )
DEM_pd['x'] = DEM_pd['x'] * x_in_orig
DEM_pd['y'] = DEM_pd['y'] * y_in_orig
DEM_pd['z'] = DEM_pd['z'] * relief
DEM_pd['x'] = DEM_pd['x'].astype(int) 
DEM_pd['y'] = DEM_pd['y'].astype(int) 

# Flip domain for correct flow direction:
DEM_pd['x'] = DEM_pd['x'].values[::-1]

print("\nNormalized data + size: ", DEM_pd.size, "shape: ", DEM_pd.shape)
print("-----------------------------------------------------")
print(DEM_pd)

# Convert to numpy array, and transform to a 2d-array / matrix:
DEM = DEM_pd.pivot(index='y', columns='x', values='z').to_numpy()

print("\nReshaped(pivot) data + size: ", DEM.size, " shape: ", DEM.shape)
print("-----------------------------------------------------")
 
# Remove outer section (cut_m) from each. This was to remove a section that gave simulation errors.
# Can also be used to reduce domain size.

#cut_m = 20                                    # Meters to cut from each side (MINIMUM 20 and use resolution interval)
actual_input_resolution = 20
#cut_n = int(cut_m/actual_input_resolution)    # Number of data points or nodes to cut from each side (num of rows and columns)

# number of nodes to remove from each side of domain:
cut_n_x = 100
cut_n_y = 50

print("CUT x:", cut_n_x , "\nCUT y:", cut_n_y)
print("\nSize: ", DEM.size, " shape: ", DEM.shape)
DEM = DEM[cut_n_y:-cut_n_y , cut_n_x:-cut_n_x]   
print("Size: ", DEM.size, "  shape: ", DEM.shape)

original_domain_size_x = (x_in_orig - cut_n_x*2)*actual_input_resolution
original_domain_size_y = (y_in_orig - cut_n_y*2)*actual_input_resolution


#x_in_orig = 500 
#y_in_orig = 250
taper_length = 50                   #(m) new taper width
taper_height = 25                   #(m) new taper hight of y-direction taper



# Create taper DEM
left_block = np.fliplr(DEM[:, :taper_length])               #Duplicates and flips a block (l) with the width of taper_length
right_block = np.fliplr(DEM[:, -taper_length:])             #Duplicates and flips a block (r) with the width of taper_length

DEM_out = np.hstack((left_block, DEM, right_block))         #Stacks the blocks on the sides of the DEM  
plt.imshow(DEM_out)
if show_plots: plt.show()


taper_DEM_area = np.block([
    DEM_out[:,-int(taper_length/resolution):], 
    DEM_out, 
    DEM_out[:,0:int(taper_length/resolution)]
])

plt.imshow(taper_DEM_area)
plt.title("First stack fin")
if show_plots: plt.show()


# Obtain taper windows 
window_start = [
    2*min(int(taper_length/(resolution*2)), 
    int(taper_length/resolution) - xx) for xx in range(int(taper_length/resolution))]

# Taper left side:
for yy in range(np.shape(DEM_out)[0]):
    for xx in range(int(taper_length/resolution)):
        taper_ind = xx + int(taper_length/resolution)                                           #gets index
        window = window_start[xx]                                                               #sets the size (width) of taper window
        mean_ind = np.linspace(taper_ind - window/2, taper_ind + window/2, num = window)        #defines the indexes ... ?
        mean_ind = [int(ii) for ii in mean_ind]                                                 #Convert to int
        DEM_out[yy,xx] = np.mean(taper_DEM_area[yy,mean_ind])


#flip and run loop again
DEM_out = np.fliplr(DEM_out)
taper_DEM_area = np.fliplr(taper_DEM_area)

# Taper Right side:
for yy in range(np.shape(DEM_out)[0]):
    for xx in range(int(taper_length/resolution)):
        taper_ind = xx + int(taper_length/resolution) 
        window = window_start[xx] 
        mean_ind = np.linspace(taper_ind - window/2, taper_ind + window/2, num = window)
        mean_ind = [int(ii) for ii in mean_ind]
        DEM_out[yy,xx] = np.mean(taper_DEM_area[yy,mean_ind])

#flip back
DEM_out = np.fliplr(DEM_out)
taper_DEM_area = np.fliplr(taper_DEM_area)


plt.subplot(2,1,1)
plt.imshow(DEM_out)
plt.title("LR-Tapered DEM")
plt.subplot(2,1,2)
plt.imshow(DEM)
plt.title("Old DEM")
if show_plots: plt.show()

#Move x-dir taper section to down-slope side of domain:
DEM_out = np.hstack((DEM_out[:, taper_length:], DEM_out[:, :taper_length]))

#Taper in y-direction:

upper_block = np.flipud(DEM_out[:taper_height, :])             #Duplicates and flips a block (l) with the width of taper_length
lower_block = np.flipud(DEM_out[-taper_height:, :])            #Duplicates and flips a block (r) with the width of taper_length

DEM_out_vertical = np.vstack((upper_block, DEM_out, lower_block))         #Stacks the blocks on the top/bottom of the DEM  
plt.imshow(DEM_out_vertical)
if show_plots: plt.show()

#add extra taper length blocks for smoothing:
taper_DEM_area_vertical = np.vstack((lower_block, DEM_out_vertical, upper_block))         #Stacks the blocks on the top/bottom of the DEM again
plt.imshow(taper_DEM_area_vertical)
plt.title("Vertical taper stack")
if show_plots: plt.show()

# Obtain taper windows 
window_start = [
    2*min(int(taper_height/(resolution*2)), 
    int(taper_height/resolution) - yy) for yy in range(int(taper_height/resolution))]

# Taper left side:
for xx in range(np.shape(DEM_out_vertical)[1]):
    for yy in range(int(taper_height/resolution)):
        taper_ind = yy + int(taper_height/resolution) 
        window = window_start[yy] 
        mean_ind = np.linspace(taper_ind - window/2, taper_ind + window/2, num = window)
        mean_ind = [int(ii) for ii in mean_ind]
        DEM_out_vertical[yy, xx] = np.mean(taper_DEM_area_vertical[mean_ind, xx])


#flip and run loop again
DEM_out_vertical = np.flipud(DEM_out_vertical)
taper_DEM_area_vertical = np.flipud(taper_DEM_area_vertical)

# Taper lower side:
for xx in range(np.shape(DEM_out_vertical)[1]):
    for yy in range(int(taper_height/resolution)):
        taper_ind = yy + int(taper_height/resolution) 
        window = window_start[yy] 
        mean_ind = np.linspace(taper_ind - window/2, taper_ind + window/2, num = window)
        mean_ind = [int(ii) for ii in mean_ind]
        DEM_out_vertical[yy, xx] = np.mean(taper_DEM_area_vertical[mean_ind, xx])

#flip back
DEM_out_vertical = np.flipud(DEM_out_vertical)
taper_DEM_area_vertical = np.flipud(taper_DEM_area_vertical)

#Move y-dir taper section to down-slope side of domain:
DEM_out_vertical = np.vstack((DEM_out_vertical[taper_height:, :], DEM_out_vertical[:taper_height, :]))

plt.imshow(DEM_out_vertical)
plt.title("DEM_out_vertical")
if show_plots: plt.show()

check_equal_boundaries(DEM_out_vertical)

# -----------------------------------------------------------------------------------------------------------------------
# Gaussian filter:


DEM_out = DEM_out_vertical

# Stacking 3x3 grid of DEM together, for best periodic gaussian application:
DEM_out3 = np.block([DEM_out, DEM_out, DEM_out]) # Stack 3 next to each other 
DEM_out3x3 = np.vstack([DEM_out3, DEM_out3, DEM_out3]) # Stack 3 on top of each other
plt.imshow(DEM_out3x3)
plt.title("DEM 3x3 STACK")
if show_plots: plt.show()

DEM_g3x3 = ndimage.gaussian_filter(DEM_out3x3, sigma=(g_sd, g_sd)) #apply filter to stacked periodic DEM
DEM_g_middle = DEM_g3x3[: , DEM_out.shape[1]-1:(DEM_out.shape[1]*2)] #then select the middle section
DEM_g_center = DEM_g_middle[DEM_out.shape[0]-1:(DEM_out.shape[0]*2), :] #then select the vertical middle section (center)
DEM_g = DEM_g_center

plt.imshow(DEM_g_center)
plt.title("CENTER of gaussion filetered DEM")
if show_plots: plt.show()

#check boundary
check_equal_boundaries_visual(DEM_g)

# Compare before and after Gaussian filter:
print("----------------Gaussian filter applied ---------------")
print("Shape (gaus): ", np.shape(DEM_g))
print("Shape (old): ", np.shape(DEM_out))


relief_final = abs(np.min(DEM_g)-np.max(DEM_g))



plt.subplot(2,1,1)
plt.imshow(DEM_g, cmap='terrain')
plt.colorbar() 
plt.title("Gaussian filter applied")
plt.subplot(2,1,2)
plt.imshow(DEM_out, cmap='terrain')
plt.colorbar()
plt.title(" Older DEM ")
if show_plots: plt.show()

# Plot profile through the middle of domain: 
mid_g = get_middle_profile_vector(DEM_g)
mid_old = get_middle_profile_vector(DEM_out)
plt.subplot(2,1,1)
plt.plot(mid_g)
plt.title("Middle profile after")
plt.subplot(2,1,2)
plt.plot(mid_old)
plt.title("Middle profile before")
if show_plots: plt.show()

#hillshde
DEM_hs = hillshade(DEM_g, 125, 45)

#gradient
x_gradient = np.gradient(DEM_g)[0]
y_gradient = np.gradient(DEM_g)[1]
      
#plots
fig, ax1 = plt.subplots(3,1)

im = ax1[0].imshow(DEM_g, cmap='terrain') 
cbar = plt.colorbar(im, ax=ax1[0], shrink=0.5)
cbar.set_label('Elevation, m', rotation=270, labelpad=20)
#ax1.imshow(DEM_hs, cmap='Greys', alpha=0.5)
ax1[0].set_xlabel("Along flow (m)")
ax1[0].set_ylabel("Across flow (m)")

im = ax1[1].imshow(x_gradient, cmap='terrain') 
cbar = plt.colorbar(im, ax=ax1[1], shrink=0.5)
cbar.set_label('y gradient', rotation=270, labelpad=20)
#ax1.imshow(DEM_hs, cmap='Greys', alpha=0.5)
ax1[1].set_xlabel("Along flow (m)")
ax1[1].set_ylabel("Across flow (m)")

im = ax1[2].imshow(y_gradient, cmap='terrain') 
cbar = plt.colorbar(im, ax=ax1[2], shrink=0.5)
cbar.set_label('x gradient', rotation=270, labelpad=20)
#ax1.imshow(DEM_hs, cmap='Greys', alpha=0.5)
ax1[2].set_xlabel("Along flow (m)")
ax1[2].set_ylabel("Across flow (m)")
if show_plots: plt.show() 


#final check boundary
print("\nPERIODICITY TEST: ")
left_right, top_bottom = check_equal_boundaries(DEM_g)
if left_right and top_bottom:
    print("\nIt is xy-periodic!!!")
    print("Size of original DEM used (before taper): x:", original_domain_size_x,"(m), y:", original_domain_size_y, "(m)" )
    print("Final Shape: ", np.shape(DEM_g))
    print("Final relief (m): ", relief_final)

elif not left_right:
    print("Left and right sides are not matching")
elif not top_bottom:
    print("Top and bottom not matching")
else: 
    print("something is wrong with the boundaries")


# Save as .npy for use in "npy_to_xyz_simple.m"
np.save(path_output, DEM_g)
print("\nSaved as: ", path_output)
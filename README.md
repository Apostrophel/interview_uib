# DEM Processing for Elmer/Ice
sjurbarndon@proton.me

These scripts handle .tif and .scv datasets and ultimately outputs a .dat file for use with Elmer/Ice. Process elevation data after creating a complete raster of the domain saved in a .tif file (Using QGIS). Rotation might be required before processing to align the domain with the ice flow direction.


### Script List
1. dem_conversion_taper_xy.py
2. npy_to_xyz.m
3. rename_columns.py                
4. rotate_tif_files.py

### Steps
 
1. Find the angle of rotation (...by guessing) and run the script 
    rotate_tif_files.py specifying the angle as rotation angle.

2. Open the new, rotated raster in QGIS for sampling and export the data as a .csv file: 
   1. Create "Regular Points" over the raster in desired resolution.
   2. Run "Sample Raster Values" 
   3. Save the resulting points as a .scv

3. The sampled domain (.csv) can be loaded in the script (dem_conversion_taper_xy.py) for tapering and periodicity in both direction. This will output a .npy file for use in the script (npy_to_xyz_simple.m). Record the terminal output of dem_conversion_taper_xy.py for use in next steps. Specifically "Final Shape: (xxx, yyy)"  and "Final relief (m): xxx.xx"
    
4. Run npy_to_xyz.m with the resulting .npy above.
    Here you specify ice thickness by setting the "ice_thickness" input variable. Ice thicknes will be calculated from highest point in topography (plataeu).
    Specify relief input variable using "Final relief (m): ..." from prevoius step.
    Specify dem dimentions (d_x and d_y) using "Final Shape: " from prevoius step. Remember to add 1 to each value.

5. Use the final resulting .dat file in Elmer/Ice  (Check if the dat-file works with enth_output.sif)

    IMPORTANT NOTE: 
        Line 1 in the dat should look like this: 0 0 (some z value)
    
    IMPORTANT NOTE II: 
        dat file is saved with the format yxz instead of xyz and this needs to be specified
        in the Elmer sif file under the Read DEM Solver (Procedure = "ElmerIceSolvers" "Grid2DInterpolator"):   
            Variable 1 Invert = Logical True !this switches x and y order requirements

    IMPORTANT NOTE III: 
        Remember  "Final Shape: (..., ...)" from dem_conversion_taper_xy.py terminal output will be used in 
        the Elmer .sif file, AND when creating a mesh for Elmer. The domains actual length and width in
        meters is also necessary.
        
        For some reason there is an extra line for each dimention in the resulting .dat file. So in the sif,
        use final shape + 1 for each value.








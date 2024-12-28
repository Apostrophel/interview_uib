% Script to take .npy output from dem_conversion_taper_xy.py
% and make an Elmer/Ice readable .xyz DEM.
% Sjur Barndon


% Input dimensions
ice_thickness = 2000; % This will be final ice thickness above dem (plateau) elevation
relief = 1234.3946680220822; %(m) relief of domain (See output of dem_conversion_taper_xy.py script)
depth = relief + ice_thickness;

% (d_x, d_y, and res should match DEM dimensions, and will need to be kept up to date in Elmer)
res = 20; %(m) x and y resolution
d_x = 401*res; %(m) total length of domain (xnodes * res)
d_y = 201*res; %(m) total width of domain  (ynodes * res)

% Change working directory to script location
currentScriptPath = mfilename('fullpath');
[currentScriptDir, ~, ~] = fileparts(currentScriptPath);
cd(currentScriptDir);

% Add .npy reader
addpath('/home/sjur/Documents/gitrepos/npy-matlab/npy-matlab');

% Load .npy to matlab array
DEM = readNPY('veafjorden_topography_r20_8k_4k_shift.npy');
filename_save = 'veafjorden_8k_4k_d2000_shift.dat';

size(DEM)
DEM = DEM - depth;

max(DEM, [], 'all')-min(DEM, [], 'all');

figure(2)
imagesc(DEM)
colorbar

% Create x_grid and y_grid
x_grid = repmat((0:res:d_x), d_y/res + 1, 1);
y_grid = repmat((0:res:d_y)', 1, d_x/res + 1);
z_grid = DEM;

size(x_grid)
size(y_grid)

% Make .xyz
fprintf('Size of incoming .npy DEM is %4.1f y by %4.1f x', size(DEM, 1), size(DEM, 2))

% Save data as xyz
out = zeros(size(z_grid,1)*size(z_grid,2),3);

% Grid to xyz
for i = 1:size(z_grid, 2) %x
    for j = 1:size(z_grid, 1) %y
        out((i-1)*((d_y/res)+1) + j,1) = x_grid(1,i); %set x value
        out((i-1)*((d_y/res)+1) + j,2) = y_grid(j,1); %set y value
        out((i-1)*((d_y/res)+1) + j,3) = z_grid(j,i); %set z value       
    end
end

% Sort and remove all lines with the value 0 0 0:
out = sortrows(out);
out = out(~all(out == [0 0 0], 2), :);  


%_______________________TESTING____________________________________________________
fprintf('\nCheck first 10 lines: \n');
fprintf('%f %f %f\n', out(1:10, :)');
relief;

%check boundaries:
left_edge_rows = out(out(:, 1) == 0, :);            %rows_with_first_column_zero 
right_edge_rows = out(out(:, 1) == d_x-res, :);     %rows_with_first_column_16000 

top_edge_rows = out(out(:, 2) == 0, :);             %rows_with_second_column_zero 
bottom_edge_rows = out(out(:, 2) == d_y-res, :);    %rows_with_second_column_9000

% Loop through the extracted rows and print "z" values
for i = 1:10
    left_z = left_edge_rows(i, 3);
    right_z = right_edge_rows(i, 3);
    fprintf(['sides %d: %.6f  =?= %.6f \n'], i, left_z, right_z);
end

for i = 1:10
    top_z = top_edge_rows(i, 3);
    bot_z = bottom_edge_rows(i, 3);
    fprintf(['top/bot %d: %.6f  =?= %.6f \n'], i, top_z, bot_z);
end
%______________________TEST DONE___________________________________________________


% Save .dat file
dlmwrite(filename_save,out,'delimiter',' ')

% Visual plotting
figure(3)
ax = surf(DEM, 'LineStyle', 'none');
daspect([1 1 1])
lighting gouraud;
colormap default; %or copper
set(gca,'visible','off')
set(gca,'xtick',[])
set(gcf,'color','none')
set(gca,'color','none')
view([30 25])
ax.EdgeColor = 'none';

% Save the plot
%print(gcf, 'bed.png', '-dpng', '-r1500')
%saveas(gcf, 'bed.png');
%export_fig bed_simulation.png -transparent -m5 %<- this for transparent export
























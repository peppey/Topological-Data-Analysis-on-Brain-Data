label = 0

%brain_imaging_data = load('./Data/m292/run0'+string(label)+'/run0'+string(label)+'.mat');

%gcamp = brain_imaging_data.gcamp

% Later also use other fields
%brain_imaging_data = rmfield(time_series_data,"tform1") 
%brain_imaging_data = rmfield(time_series_data,"gcamp")

h5create('Brain_Imaging_Data.h5', '/Data', size(gcamp))
h5write('Brain_Imaging_Data.h5', '/Data', gcamp)

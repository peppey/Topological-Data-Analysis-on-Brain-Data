csvData = readmatrix('Label_5_data.csv');

EEG = csvData(:, 3)

EEG(1,:) = [];

% Save the data as a .mat file
save('Label_5_data.mat', 'EEG');



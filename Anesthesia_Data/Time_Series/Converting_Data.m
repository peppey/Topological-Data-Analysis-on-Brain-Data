
time_series_data = load('./Data/m292/trimm_eeg.mat');

time_series_data = rmfield(time_series_data,"finish")
time_series_data = rmfield(time_series_data,"start")
time_series_data = rmfield(time_series_data,"trig")


%csvwrite('Anesthesia_Time_Series_Data.csv', time_series_data);

writetable(struct2table(time_series_data), './Data/m292/Time_Series_Data.csv')


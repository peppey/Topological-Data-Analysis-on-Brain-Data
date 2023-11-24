
data1 = load('Label_1_data.mat');
converted_data1 = struct2array(data1);


data3 = load('Label_3_data.mat');
converted_data3 = struct2array(data3);


data5 = load('Label_1_data.mat');
converted_data5 = struct2array(data5);



f = figure(1)
f.Position = [0,0,1200,800]
cwt(converted_data1, 'amor', milliseconds(1))
set(gca, 'FontSize', 18, 'FontWeight', 'bold')


f = figure(2)
f.Position = [0,0,1200,800]
cwt(converted_data3, 'amor', milliseconds(1))
set(gca, 'FontSize', 18, 'FontWeight', 'bold')


f = figure(3)
f.Position = [0,0,1200,800]
cwt(converted_data5, 'amor', milliseconds(1))
set(gca, 'FontSize', 18, 'FontWeight', 'bold')


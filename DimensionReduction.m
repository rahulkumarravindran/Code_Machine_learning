%Reading the data from the dataset csv file
data=readmatrix("D:\Windsor\Fourth semester\Applied Machine learning\Project\Datasets\Datasets\D13.csv");

%Getting the size of the data matrix
[m,n]= size(data);

%fill missing values with median of nieghbours
data_f = fillmissing(data,'constant',0);

%remove columns filled with zeros
data_f = data_f(:,any(data_f));

%Getting the size of the processed data matrix
[m,n]= size(data_f);

%Dimension Reduction using CCA
[mappedData,mapping]= compute_mapping(data_f(:,1:n-1),"CCA",3,3,'JDQR');
%[mappedData_label,mapping] = compute_mapping(data_f(:,1:n-1),"MVU",2,3);
   
%Writing the data to a CSV
writematrix(mappedData,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D13_CE.csv","Delimiter","comma");
disp("The dimension reduced (Conformal Eigenmaps) dataset has been written to a CSV file")

%Dimension Reduction using MVU
[mappedData,mapping] = compute_mapping(data_f(:,1:n-1),"MVU",3,3,'JDQR');

%Writing the data to csv
writematrix(mappedData,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D13_MVU.csv","Delimiter","comma");
disp("The dimension reduced (Maximum Variance Unfolding) dataset has been written to a CSV file")

%Dimension Reduction using Landmark-MVU
[mappedData,mapping] = compute_mapping(data_f(1:3300,1:n-1),"LandmarkMVU",1,2,'JDQR');

%Writing the data to csv
writematrix(mappedData,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D13_LMVU.csv","Delimiter","comma");
disp("The dimension reduced (Landmark Maximum Variance Unfolding) dataset has been written to a CSV file")




%Reading the data from the dataset csv file
data=readmatrix("D:\Windsor\Fourth semester\Applied Machine learning\Project\Datasets\Datasets\D5.csv");
data=data(1:50,:);

%Getting the size of the data matrix
[m,n]= size(data);

%fill missing values with median of nieghbours
data_f = fillmissing(data,'constant',0);

%remove columns filled with zeros
data_f = data_f(:,any(data_f));

%Getting the size of the processed data matrix
[m,n]= size(data_f);

%Dimension Reduction using Conformal Eigenmaps
[mappedData,mapping]= compute_mapping(data_f(:,1:n-1),"CCA",5,2);

%some training examples may be ignored because there are not nearest
%neighbors (nij is 1 for nearest neighbors and 0 for others)
%the indexes of the data points that are k-nearest neighbors are stored in
%mapping.conn_comp
corres_labels=data_f(mapping.conn_comp,n);
[m_mapped,n_mapped] = size(mappedData);
mappedData(:,n_mapped+1)=corres_labels;


%Writing the data to a CSV
writematrix(mappedData,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D5_CE.csv","Delimiter","comma");
disp("The dimension reduced (Conformal Eigenmaps) dataset has been written to a CSV file")

%Getting the size of the processed data matrix
[m,n]= size(data_f);

%Dimension Reduction using MVU
[mappedData,mapping] = compute_mapping(data_f(:,1:n-1),"MVU",5,2);

%some training examples may be ignored because there are not nearest
%neighbors (nij is 1 for nearest neighbors and 0 for others)
%the indexes of the data points that are k-nearest neighbors are stored in
%mapping.conn_comp
corres_labels=data_f(mapping.conn_comp,n);
[m_mapped,n_mapped] = size(mappedData);
mappedData(:,n_mapped+1)=corres_labels;

%Writing the data to csv
writematrix(mappedData,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D5_MVU.csv","Delimiter","comma");
disp("The dimension reduced (Maximum Variance Unfolding) dataset has been written to a CSV file")

%preprocessing data for LMVU
TotalOutputRows=1000;
data_f=sortrows(data_f,n);
data_LMVU=[];
labels_unique=unique(data_f(:,n));
noOfLabels=length(labels_unique);
RowsPerLabel=round(TotalOutputRows/noOfLabels);
for i=1:noOfLabels
    temp_list=data_f(data_f(:,n)==labels_unique(i),:);
    [m_temp,n_temp]=size(temp_list);
    [m_LMVU,n_LMVU]=size(data_LMVU);
    if RowsPerLabel<m_temp
        %data_LMVU(m_LMVU+1:m_LMVU+RowsPerLabel+1,:)=temp_list(1:RowsPerLabel,:);
        data_LMVU=[data_LMVU;temp_list(1:RowsPerLabel,:)];
    else
        %data_LMVU(m_LMVU+1:m_LMVU+m_temp+1,:)=temp_list(:,:);
        data_LMVU=[data_LMVU;temp_list];
    end
end

%Dimension Reduction using Landmark-MVU
[mappedData,mapping] = compute_mapping(data_LMVU(:,1:n-1),"LandmarkMVU",5,2);

%Adding the corresponding labels to the dimension reduced data
[m_mapped,n_mapped] = size(mappedData);
mappedData(:,n_mapped+1)=data_LMVU(:,n);

%Writing the data to csv
writematrix(mappedData,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D5_LMVU.csv","Delimiter","comma");
disp("The dimension reduced (Landmark Maximum Variance Unfolding) dataset has been written to a CSV file")




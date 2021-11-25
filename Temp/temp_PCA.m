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

d=24;
iter=round(d/3);
batch_size=round(n/iter)-1;
idx=[];

for i=1:iter
    idx(i)=i*batch_size;
end

for i=1:iter
    start_idx=(i-1)*batch_size+1;
    end_idx=idx(i);
    
    %Dimension Reduction using Conformal Eigenmaps
    [mappedData,mapping]= compute_mapping(data_f(:,start_idx:end_idx),"CCA",3,2);

    %some training examples may be ignored because there are not nearest
    %neighbors (nij is 1 for nearest neighbors and 0 for others)
    %the indexes of the data points that are k-nearest neighbors are stored in
    %mapping.conn_comp
    corres_labels=data_f(mapping.conn_comp,n);
    [m_mapped,n_mapped] = size(mappedData);
    mappedData(:,n_mapped+1)=corres_labels;
    [m_mapped,n_mapped] = size(mappedData);
    [m_transformed,n_transformed] =size(transformed_matrix);
    
    if(m_transformed==0)
        m_transformed=m_mapped;
    end
    
    if(m_transformed>m_mapped)
        transformed_matrix=[transformed_matrix(1:m_mapped,:) mappedData(:,:)];
    else
        transformed_matrix=[transformed_matrix(:,:) mappedData(1:m_transformed,:)];
    end
end

%Writing the data to a CSV
writematrix(transformed_matrix,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D13_CE.csv","Delimiter","comma");
disp("The dimension reduced (Conformal Eigenmaps) dataset has been written to a CSV file")

transformed_matrix=[];

for i=1:iter
    start_idx=(i-1)*batch_size+1;
    end_idx=idx(i);
    
    %Dimension Reduction using Conformal Eigenmaps
    [mappedData,mapping]= compute_mapping(data_f(:,start_idx:end_idx),"MVU",3,2);

    %some training examples may be ignored because there are not nearest
    %neighbors (nij is 1 for nearest neighbors and 0 for others)
    %the indexes of the data points that are k-nearest neighbors are stored in
    %mapping.conn_comp
    corres_labels=data_f(mapping.conn_comp,n);
    [m_mapped,n_mapped] = size(mappedData);
    mappedData(:,n_mapped+1)=corres_labels;
    [m_mapped,n_mapped] = size(mappedData);
    [m_transformed,n_transformed] =size(transformed_matrix);
    
    if(m_transformed==0)
        m_transformed=m_mapped;
    end
    
    if(m_transformed>m_mapped)
        transformed_matrix=[transformed_matrix(1:m_mapped,:) mappedData(:,:)];
    else
        transformed_matrix=[transformed_matrix(:,:) mappedData(1:m_transformed,:)];
    end
end

%Writing the data to a CSV
writematrix(transformed_matrix,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D13_MVU.csv","Delimiter","comma");
disp("The dimension reduced (Conformal Eigenmaps) dataset has been written to a CSV file")

transformed_matrix=[];

for i=1:iter
    start_idx=(i-1)*batch_size+1;
    end_idx=idx(i);
    
    %Dimension Reduction using Conformal Eigenmaps
    [mappedData,mapping]= compute_mapping(data_f(:,start_idx:end_idx),"LandmarkMVU",1,2);

    %some training examples may be ignored because there are not nearest
    %neighbors (nij is 1 for nearest neighbors and 0 for others)
    %the indexes of the data points that are k-nearest neighbors are stored in
    %mapping.conn_comp
    corres_labels=data_f(mapping.conn_comp,n);
    [m_mapped,n_mapped] = size(mappedData);
    mappedData(:,n_mapped+1)=corres_labels;
    [m_mapped,n_mapped] = size(mappedData);
    [m_transformed,n_transformed] =size(transformed_matrix);
    
    if(m_transformed==0)
        m_transformed=m_mapped;
    end
    
    if(m_transformed>m_mapped)
        transformed_matrix=[transformed_matrix(1:m_mapped,:) mappedData(:,:)];
    else
        transformed_matrix=[transformed_matrix(:,:) mappedData(1:m_transformed,:)];
    end
end

%Writing the data to a CSV
writematrix(transformed_matrix,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D13_LMVU.csv","Delimiter","comma");
disp("The dimension reduced (Conformal Eigenmaps) dataset has been written to a CSV file")

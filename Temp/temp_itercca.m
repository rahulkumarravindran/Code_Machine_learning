%Reading the data from the dataset csv file
data=readmatrix("D:\Windsor\Fourth semester\Applied Machine learning\Project\Datasets\Datasets\D1.csv");

%Getting the size of the data matrix
[m,n]= size(data);

%fill missing values with median of nieghbours
data_f = fillmissing(data,'constant',0);

%remove columns filled with zeros
data_f = data_f(:,any(data_f));

%Getting the size of the processed data matrix
[m,n]= size(data_f);

d=12;
iter=round(d/3);
batch_size=round(n/iter)-1;
idx=[];

for i=1:iter
    idx(i)=i*batch_size;
end

%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%
%Conformal Eigenmaps
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%


transformed_matrix=[];

for i=1:iter
    start_idx=(i-1)*batch_size+1;
    end_idx=idx(i);
    
    %Dimension Reduction using Conformal Eigenmaps
    [mappedData,mapping]= compute_mapping(data_f(:,start_idx:n-1),"CCA",3,2);

    %some training examples may be ignored because there are not nearest
    %neighbors (nij is 1 for nearest neighbors and 0 for others)
    %the indexes of the data points that are k-nearest neighbors are stored in
    %mapping.conn_comp
    
    [m_mapped,n_mapped] = size(mappedData);
    [m_transformed,n_transformed] =size(transformed_matrix);
    %disp(m_transformed)
    %disp(m_mapped)
    
    
    
    if(m_transformed==0)
        m_transformed=m;
    end
    
    if(m_transformed>m_mapped)
        eigen=zeros(abs(m_transformed-m_mapped),3);
        [m_mapped,n_mapped] = size([mappedData(:,:);eigen]);
        [m_transformed,n_transformed] =size(transformed_matrix);
        %disp("1")
        %disp(m_mapped)
        %disp(m_transformed)
        transformed_matrix=[transformed_matrix(:,:) [mappedData(:,:);eigen]];
    else
        
        eigen=zeros(abs(m_transformed-m_mapped),3*i);
        
        [m_mapped,n_mapped] = size(mappedData);
        [m_eigen,n_eigen]=size(eigen);
        [m_transformed,n_transformed] =size(transformed_matrix);
        %disp("2")
        %disp(n_eigen)
        %disp(n_transformed)
        
        transformed_matrix=[[transformed_matrix(:,:);eigen] mappedData(1:m_transformed,:)];
    end
    
    max=0;
    label=[];
    if(max<length(mapping.conn_comp))
        max=length(mapping.conn_comp);
        label=mapping.conn_comp;
    end
end

[m_transformed,n_transformed] =size(transformed_matrix);

corres_labels=data_f(label,n);

[m_labels,n_labels] =size(corres_labels);

temp=zeros(abs(m_transformed-m_labels),1);

transformed_matrix(:,n_transformed+1)=[corres_labels;temp];

%Writing the data to a CSV
writematrix(transformed_matrix,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D1_CE.csv","Delimiter","comma");
disp("The dimension reduced (Conformal Eigenmaps) dataset has been written to a CSV file")

%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%
%Maximum Variance unfolding
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%


transformed_matrix=[];

for i=1:iter
    start_idx=(i-1)*batch_size+1;
    end_idx=idx(i);
    
    %Dimension Reduction using Conformal Eigenmaps
    [mappedData,mapping]= compute_mapping(data_f(:,start_idx:n-1),"MVU",3,2);

    %some training examples may be ignored because there are not nearest
    %neighbors (nij is 1 for nearest neighbors and 0 for others)
    %the indexes of the data points that are k-nearest neighbors are stored in
    %mapping.conn_comp
    
    [m_mapped,n_mapped] = size(mappedData);
    [m_transformed,n_transformed] =size(transformed_matrix);

    if(m_transformed==0)
        m_transformed=m;
    end
    
    if(m_transformed>m_mapped)
        eigen=zeros(abs(m_transformed-m_mapped),3);
        [m_mapped,n_mapped] = size([mappedData(:,:);eigen]);
        [m_transformed,n_transformed] =size(transformed_matrix);
        %disp("1")
        %disp(m_mapped)
        %disp(m_transformed)
        transformed_matrix=[transformed_matrix(:,:) [mappedData(:,:);eigen]];
    else
        
        eigen=zeros(abs(m_transformed-m_mapped),3*i);
        
        [m_mapped,n_mapped] = size(mappedData);
        [m_eigen,n_eigen]=size(eigen);
        [m_transformed,n_transformed] =size(transformed_matrix);
        %disp("2")
        %disp(n_eigen)
        %disp(n_transformed)
        
        transformed_matrix=[[transformed_matrix(:,:);eigen] mappedData(1:m_transformed,:)];
    end
    
    max=0;
    label=[];
    if(max<length(mapping.conn_comp))
        max=length(mapping.conn_comp);
        label=mapping.conn_comp;
    end
end

[m_transformed,n_transformed] =size(transformed_matrix);

corres_labels=data_f(label,n);

[m_labels,n_labels] =size(corres_labels);

temp=zeros(abs(m_transformed-m_labels),1);

transformed_matrix(:,n_transformed+1)=[corres_labels;temp];

%Writing the data to a CSV
writematrix(transformed_matrix,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D1_MVU.csv","Delimiter","comma");
disp("The dimension reduced (Maximum Variance Unfolding) dataset has been written to a CSV file")

%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%
%Landmark Maximum Variance unfolding
%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%

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
    [m_LMVU,n_LMVU]=size(data_LMVU);
end

[m_LMVU,n_LMVU]=size(data_LMVU);

transformed_matrix=[];

for i=1:iter
    start_idx=(i-1)*batch_size+1;
    end_idx=idx(i);
    
    %Dimension Reduction using Conformal Eigenmaps
    [mappedData,mapping]= compute_mapping(data_LMVU(:,start_idx:n-1),"LandmarkMVU",2,2);

    %some training examples may be ignored because there are not nearest
    %neighbors (nij is 1 for nearest neighbors and 0 for others)
    %the indexes of the data points that are k-nearest neighbors are stored in
    %mapping.conn_comp
    
    [m_mapped,n_mapped] = size(mappedData);
    [m_transformed,n_transformed] =size(transformed_matrix);

    if(m_transformed==0)
        m_transformed=m_LMVU;
    end
    
    if(m_transformed>m_mapped)
        eigen=zeros(abs(m_transformed-m_mapped),2);
        [m_mapped,n_mapped] = size([mappedData(:,:);eigen]);
        [m_transformed,n_transformed] =size(transformed_matrix);
        %disp("1")
        %disp(m_mapped)
        %disp(m_transformed)
        transformed_matrix=[transformed_matrix(:,:) [mappedData(:,:);eigen]];
    else
        
        eigen=zeros(abs(m_transformed-m_mapped),2*i);
        
        [m_mapped,n_mapped] = size(mappedData);
        [m_eigen,n_eigen]=size(eigen);
        [m_transformed,n_transformed] =size(transformed_matrix);
        %disp("2")
        %disp(n_eigen)
        %disp(n_transformed)
        
        transformed_matrix=[[transformed_matrix(:,:);eigen] mappedData(1:m_transformed,:)];
    end
    
    max=0;
    
    label=data_LMVU(:,n_LMVU);
end

transformed_matrix(:,n_transformed+1)=label;

%Writing the data to a CSV
writematrix(transformed_matrix,"D:\Windsor\Fourth semester\Applied Machine learning\Project\Code_Machine_learning\DimensionReducedDataSet\D1_LMVU.csv","Delimiter","comma");
disp("The dimension reduced (Landmark Maximum Variance Unfolding) dataset has been written to a CSV file")

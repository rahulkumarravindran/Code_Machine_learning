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

%preprocessing data for LMVU
TotalOutputRows=3300;
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
disp(size(data_LMVU))
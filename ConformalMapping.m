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

%Splitting the data into train and test data
p=0.80;
rng(2000)
random_indices=randperm(m);
train_data = data_f(random_indices(1:round(p*m)),:);
test_data = data_f(random_indices(round(p*m)+1:end),:) ;

%size of test and train data
[m_train,n_train]=size(train_data);
[m_test,n_test]=size(test_data);

%extracting the data and the labels for both train and test data
x_train= train_data(1:m_train,1:n_train-1);
y_train= train_data(1:m_train,n_train);
x_test= test_data(1:m_test,1:n_test-1);
y_test= test_data(1:m_test,n_test);

%Conformal eigenmaps on the train dataset
[mappedData_train,mapping]=compute_mapping(x_train, 'LandmarkMVU',57,3);

%size of the dimentionality reduced matrix
[m_train,n_train]=size(mappedData_train);

%Apply conformal eigenmap DR on test data
[mappedData_test,mapping]=compute_mapping(x_test, 'CCA', 4,3);
[m_test,n_test]=size(mappedData_test);

%Classification 1.1: Naive bayes on the DR dataset
%Training the model using the training Dataset
Mdl_CE=fitcnb(mappedData_train,y_train(1:m_train,:));
disp("Model trained using Dimensionality reduced dataset")

%Calculate Loss(error) of the Model using the DR test dataset
test_loss=loss(Mdl_CE,mappedData_test,y_test(1:m_test,:));
disp("Loss for Dimension reduced test dataset");
%disp(test_loss)
CVMdl = crossval(Mdl_CE);
Loss = kfoldLoss(CVMdl)

%Classification 1.2: Naive bayes on the original dataset
%Training the model using the training Dataset
Mdl=fitcnb(x_train,y_train);
disp("Model trained using original dataset")

%Calculate Loss(error) of the Model using the original test dataset
test_loss=loss(Mdl,x_test,y_test);
disp("Loss for original test dataset");
%disp(test_loss);
CVMdl = crossval(Mdl);
Loss = kfoldLoss(CVMdl)







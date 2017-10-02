function [new_annotations, min_comp] = pcaClassifier(training_data,training_annotations, validation_data,m)
%Define storage matrix for norms of difference of approximation and individual training_data vectors
X_storage = [];
%Partition training_data into X(j) according to training_annotations
unique_training_annotations = unique(training_annotations); %list of category names
for j = 1:length(unique_training_annotations) %j is our category number
    index = unique_training_annotations(j);
    I_j = find(training_annotations == index); %Which training_annotationss line up with which observations
    training_data_j = training_data(:,I_j);
    
    %For each X(j) compute SVD
    [U ,D, V ] = svd(training_data_j);
    %Compute P_j
    P_j = U(:,1:m) * U(:,1:m)'; %rank(P_j) == m, m x m
    X_approx = P_j * validation_data;
    
    for i = 1:size(validation_data, 2)
        X_storage(j,i) = norm( validation_data(:,i) - X_approx(:,i), 'fro'); %Norm of difference of approxmation and individual training_data vectors
    %Each row represents a class training_annotations 
    end
end

[min_comp, f_x] = min(X_storage); %f_x is our PCA classifier
new_annotations = unique_training_annotations(f_x); %Match category number with right category name

end
function [new_annotations] = ldaClassifier(train,annotations, test,m)
%Compute the LDA components of each submatrix in the training set
[~, Q] = lda(train, annotations, m); %Q is Q transposed, n by m
unique_annotations = unique(annotations); %list of category names
for j = 1:length(unique_annotations) %j is our category number
    index = unique_annotations(j);
    I_j = find(annotations == index); %Which annotationss line up with which observations
    train_j = train(:,I_j);
    Z_j = Q' * train_j; % m by pj
    c_j = (1/size(Z_j, 2)) * sum(Z_j, 2);  
    
    test_proj = Q' * test; %m by p_test; our LDA projection (made by training data) onto test data
    
    for i = 1:size(test, 2)
        X_storage(j,i) = norm( test_proj(:,i) - c_j, 'fro'); %Norm of difference of approxmation and individual training_data vectors
    %Each row represents a class training_annotations 
    end
end

[~, f_x] = min(X_storage); %f_x is our LDA classifier
new_annotations = unique_annotations(f_x); %Match category number with right category name

end
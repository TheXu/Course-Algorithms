function [Y,q] = lda(data,annotation,k)
%Y is projected data
%q is LDA directions
c = (1 / size(data, 2)) * sum(data, 2); %Global Centroid 
S_w(1:size(data,1), 1:size(data,1)) = 0; %Initialize S_w; Within Cluster Scatter matridata
S_b(1:size(data,1), 1:size(data,1)) = 0; %Initialize S_b; Between Cluster Scatter matridata
unique_annotation = unique(annotation);
for ell = 1:length(unique_annotation)
    index = unique_annotation(ell);
    I_ell = find(annotation == index); %Which annotations line up with which observations
    data_ell = data(:,I_ell); %Vectors in the cluster
    c_ell = (1/size(data_ell, 2)) * sum(data_ell, 2); %Centroid of the cluster aka local centroid
    data_ell_c = data_ell - c_ell * ones(1, size(data_ell, 2));
    S_ell = data_ell_c * data_ell_c'; %Scatter matridata for ell-th cluster
    S_w = S_w + S_ell; %Within-cluster matridata added after each iteration
    
    %%%Between cluster
    S_b_ell = size(data_ell, 2) * (c_ell - c) * (c_ell - c)'; %One sum of the between cluster scatter matridata
    S_b = S_b + S_b_ell; %Between-cluster matridata added after each iteration
end
if (det(S_w) ~= 0)
    [eigenvectors, eigenvalues] = eigs(S_w \ S_b);
elseif (det(S_w) == 0)
    eps = 10^(-12);
    S_w = S_w + eps * eye(size(S_w, 2));
    [eigenvectors, eigenvalues] = eigs(S_w \ S_b);
end

eigen(:,2) = diag(eigenvalues);
eigen(:,1) = 1:size(diag(eigenvalues));
[~, order] = sort(eigen(:,2), 'descend');
q = eigenvectors(:,order(1:k)); %LDA projections are unscaled though
alpha = diag(sqrt(q(:,:)' * S_w * q(:,:))); %Find the adequate scalings for eachh lda direction
%By computing q'S_wq = 1
for i = 1:length(alpha)
    q(:,i) = q(:,i) / alpha(i); %Adjust LDA projections by the scalings
end

Y = data' * q; %Transforming the data onto the new subspace
end
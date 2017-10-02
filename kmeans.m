function [data_clustered, t, labels] = kmeans(data,k, tau, max_iter)
%Fix k, the number of clusters a tolerance tau > 0
%Randomly Initialize: Give a partitioning 
labels(1,:) = randperm(size(data,2),size(data,2)); %randi(k,size(data,2),1)' is a row of randomly generated integers from 1 to k
bins = round(size(data,2)/k); %Set bins based off of size of k
random = randperm(k);
labels(2,:) = 0; %Initialize labels 2nd row (our assignment of k labels)
for i = 1:k
    labels(2,(bins * (i - 1) + 1):(bins * i)) = random(i);
end
if size(data,2) < (bins * i)
    labels = labels(:,1:size(data,2)); %Clean off if size of labels generated are higher than the size of the data
end
%Sort the matrix but keep the column structure
[~, order] = sort(labels(1,:));
labels = labels(:,order);
label = labels(2,:);
%Set t = 0
t = 0;
%Set Qt's
Qt = Inf;
Qt_1 = 0;
while (abs(Qt_1 - Qt) > abs(tau)) && (t <= max_iter)
     Qt = Qt_1;
     Q = 0;
     %Iteration:
     for ell = 1:k %Iterating on clusters
         %Updating Step: For each cluster Il, compute the cluster centroid
         I_ell = find(label == ell);
         %I_ell = labels(1,labels(2,:) == ell); %Indices to point to the cluster
         D_ell = data(:,I_ell); %Data Points within cluster
         c_ell = (1/size(D_ell,2)) * sum(D_ell, 2); %Centroid for the cluster
         
         for j = 1:size(data,2)
           ql_val(ell, j) = norm(data(:,j) - c_ell); %For each data point, calculates the euclidean norm between
           %centroids and data point
         end
        q_ell = sum(ql_val(ell,I_ell)); %Partial Coherence
        Q = Q + q_ell;
     end
     Qt_1 = Q %Overall coherence
     %Assignment step: For each x(j), find the closest cluster centroid,
     %and assign x(j) to the corresponding cluster. This redefines the
     %partitioning, data(row_num, :)
     [~, label] = min(ql_val); %Find the cluster group(row index) that corresponds with the minimum euclidean norm between centroid
     %and data point
     t = t + 1 %Iterations
      
end

data_clustered = [data; label];
end

     
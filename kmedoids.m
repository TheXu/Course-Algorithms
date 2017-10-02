function [data_clustered, t] = kmedoids(data,k, p, dist, tau, max_iter)
%Fix k, the number of clusters a tolerance tau > 0
%Initialize: Give the initial set of characteristic vectors, or medoids,
I_m = randperm(size(data,2),k); %Pick k number of observations. They now medoids
%Set t = 0
t = 0;
%Define distance matrix
for i = 1:size(data,2)
    for j = 1:size(data,2)
        if dist == 1
            D(i, j) = norm(data(:,i) - data(:,j), p); %L1 distance
        elseif dist == 0
            [counts, ~, ~, labels] = crosstab(data(:,i),data(:,j));
            if size(counts) == [2,1]
                if labels{1,2} == '-1'
                    D(i, j) = (counts(2,1)) / (counts(2,1) + counts(1,1));
                elseif labels{1,2} == '1'
                    D(i, j) = (counts(1,1)) / (counts(2,1) + counts(1,1));
                end
            elseif size(counts) == [1,2]
                if labels{1,1} == '-1'
                    D(i, j) = (counts(1,2)) / (counts(1,2) + counts(1,1));
                elseif labels{1,1} == '1'
                    D(i, j) = (counts(1,1)) / (counts(1,2) + counts(1,1));
                end
            elseif size(counts) == [1,1]
                if labels{1,1} == labels{1,2}
                    D(i,j) = 0;
                elseif ~labels{1,1} == labels{1,2}
                    D(i,j) = 1;
                end
            elseif size(counts) == [2,2]
                D(i, j) = (counts(1,2) + counts(2,1)) / (counts(1,2) + counts(2,1) + counts(1,1) + counts(2,2));
            end
        end
    end
end

%Initialize Qt's
Qt = Inf;
Qt_1 = 0;
while (abs(Qt - Qt_1) > abs(tau)) && (t <= max_iter)
    Qt = Qt_1;
    D_m = D(:,I_m); % Distances to current medoids; Choose new medoids

    %Assignment step
    [~, I_assign] = min(D_m'); % Assignments to clusters is I_assign
    %Qt_1 = sum(q.^2) %overall coherence %Wrong
    Q = 0; %Initialize Qt_1
    I_m_temp = []; %Re-initiliaze I_m (medoid labels)
    %Update step
    for ell = 1:k
        I_ell = find(I_assign == ell); %Indices to point to the cluster
        D_ell = D(I_ell, I_ell); %Local Distance Matrix
        [~, j] = min(sum(D_ell, 1)); %q is the minimum of all possible cluster coherences, and the jth cluster element is
        %the optimal medoid, Set ql = min{sum(D_ell) | 1<=j<=pl}
        %Remember j is the jth element in the medoid group
        %sum(D_ell) is row or column sum vector
        q_ell = sum(D(I_ell, I_m(ell))); %Partial coherence of ell; I_ell is the points associated with cluster ell
        %I_m(ell) is the medoid for ell
        I_m_temp = [I_m_temp; I_ell(j)]; %Add the j into array for each iteration of ell
        Q = Q + q_ell;
    end
    I_m = I_m_temp;
    Qt_1 = Q
    t = t + 1
end

D_m = D(:,I_m); % Distances to current medoids

%Assignment step
[q, I_assign] = min(D_m'); % Assignments to clusters
data_clustered = [data; I_assign];
end

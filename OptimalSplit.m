function [j_opt, s_opt] = OptimalSplit(I, C, X)
%Finding optimal split(s_opt) on a variable(j_opt)

%I is the index vector containing the indices to the data points in the
%current rectangle

%C contains all the annotations
%X contains all the data

%Annotations corresponding to the index vector
annotations = C(I);
[~, ~, all.gini] = ClassDistr(C, I);
all.n = length(I);

%Initalize max gini index variable that stores max gini for each feature
ginis = [];
%Initialize split value variable that stores best split value for each
%feature
splits = [];
for j = 1:size(X,1)
    %Initialize s (discrete division points)
    s = [];
    %Row vector that contains the jth coordinates of the data points that
    %are in the rectangle
    x = X(j, I);
    %Sort components x in an increasing order
    [x_sort, J_sort] = sort(x, 'ascend');
    %sort corresponding annotations
    C_sort = C(I(J_sort));
    
    %Unique sorted x values
    xx = unique(x_sort);
    
    if (length(xx) > 1)
        %Define the split values to test on; the discrete division points
        for l = 1:(size(xx, 2) - 1)
            s(l) = (xx(l) + xx(l + 1)) / 2;
        end

        %Compute optimal split for each feature
        %By maximizing change in gini index
        g.delta = []; %initialize delta of gini index
        for i = 1:length(s)
            left.index = find(x_sort<s(i));
            right.index = find(x_sort>s(i));
            left.n = length(left.index);
            right.n = length(right.index);
            [~, ~, left.gini] = ClassDistr(C_sort, left.index);
            [~, ~, right.gini] = ClassDistr(C_sort, right.index);
            g.delta(i) = all.gini - ( (left.n / all.n) * left.gini ) - ( (right.n / all.n) * right.gini );
        end       
    [max_gini, split_number] = max(g.delta);
    else
        max_gini = 0; %If there are no unique values for us to split on, then we shouldn't consider
        %this feature to split on
        split_number = 1; %Doesn't matter what we pick
        s = [0];
    end
    ginis(j) = max_gini;
    splits(j) = s(split_number);
end
[~, j_opt] = max(ginis);
s_opt = splits(j_opt);

end

function [data_classified, new_labels] = treeClassifier(X, labels, test_data, test_labels)

%%%%%%%%%%%%First, make full tree with maximum depth possible
%Defining root node as a leaf
R(1).I = (1:size(X, 2));
[p, c, r] = ClassDistr(labels, R(1).I);
R(1).p = p;
R(1).j = NaN;
R(1).s = NaN;
R(1).left = NaN;
R(1).right = NaN;

%Initialize
PureNodes = [];
MixedNodes = [1]; % The root
count = 1;

%As long as there are mixed nodes we split them, putting each child either
%in the class of pure nodes or in that of mixed nodes
while length(MixedNodes) > 0
    ind = MixedNodes(1);
    I_ind = R(ind).I; %Pointer to the data points in the current rectangle
    X_ind = X(:, I_ind); % Data in the current rectangle
    [j, s] = OptimalSplit(I_ind, labels, X);
    
    %Save the optimal split values in the structure, and define pointers to
    %the children
    R(ind).j = j;
    R(ind).s = s;
    R(ind).left = count + 1; %counters to count number of splits or depth
    R(ind).right = count + 2;
    
    %Define the children as structures, figuring out which ones of the data
    %points belong to which child
    I_left = I_ind(find((X_ind(j,:)<s)));
    I_right = setdiff(I_ind, I_left);
    
    %Compute the class distribution of the data points
    if (~isempty(I_left) && ((length(unique(X(j,I_ind))) ~= 1) || (length(I_left)==1))) %Checking if I_left is not empty or if best split
        %variable isn't homogenous
        [p_left, c_left, r_left] = ClassDistr(labels, I_left);
    else 
        p_left = NaN;
        r_left = 0; %Place into PureNode for now
        I_left = NaN;
    end
    if (~isempty(I_right) && ((length(unique(X(j,I_ind))) ~= 1) || (length(I_right)==1)))
        [p_right, c_right, r_right] = ClassDistr(labels, I_right);
    else
        p_right = NaN;
        r_right = 0; %Place into PureNode for now
        I_right = NaN;
    end
    
    %Now define the data structure for the left child
    R(count + 1).I = I_left;
    R(count + 1).p = p_left;
    R(count + 1).j = NaN;
    R(count + 1).s = NaN;
    R(count + 1).left = NaN;
    R(count + 1).right = NaN;
    
    %Similarly, for the right child
    R(count + 2).I = I_right;
    R(count + 2).p = p_right;
    R(count + 2).j = NaN;
    R(count + 2).s = NaN;
    R(count + 2).left = NaN;
    R(count + 2).right = NaN;
    
    %Check if the children are pure or mixed, and add their pointers to the
    %corresponding list. Update counter
    if r_left == 0
        %The left child is a pure node
        PureNodes = [PureNodes, count + 1];
    else
        %it is a mixed node
        MixedNodes = [MixedNodes, count + 1];
    end
    
    if r_right == 0
        %The left child is a pure node
        PureNodes = [PureNodes, count + 2];
    else
        %it is a mixed node
        MixedNodes = [MixedNodes, count + 2];
    end
    
    %Delete parent node from the mixed list, update counter
    MixedNodes = MixedNodes(2:end);
    count = count + 2;
end


%%%%%%%%%%%%Second, we prune the trees
%Let's build a genealogy matrix, m x m, where m is the number of nodes in a
%tree
m = count;

%G(i,j) = 1, if and only if the node j is below the node i in the tree

%Denote the current tree as Rcurr
Rcurr = R;

%%Genealogy matrix define
%A = single step matrix
A = zeros(m,m); %Initialize with matrix full of zeroes
for k = 1:m
    i = Rcurr(k).left;
    j = Rcurr(k).right;
    if ~isnan(i)
        %the node k is not a leaf
        A(k, i) = 1;
        A(k, j) = 1; %Node k is above node i and j, else 0
    end
end

G = zeros(m, m);
for h = 1:m %Upper bound to maximum number of steps is number of nodes
    %If we reach the supremum number of steps, the number of steps
    %afterwards produces the zero matrix
    G = G + (A ^ h);
end

%Let Ileaf denote the index vector point to the leaves
Ileaf = find(sum(G, 2) == 0);

%Set Rpruned
Rpruned = Rcurr;
%Let a(1) = 0; set k = 0
k = 1;
%Initialize the pruning process, setting T1 = Tmax.
T.R(k).k = Rpruned;
T.R(k).a = 0; %Store alpha
T.R(k).G = G; %Store Genealogy Matrix
for iteration = 1:m
    %Initalize alpha from Rm. setting alpha(j) = 0 for all j
    alpha = zeros(1,m);
    no_data_points = length(Rpruned(1).I);
    for i = 1:m
        if (sum(G(i,:))==0)
            alpha(i) = Inf;
        else
            Ii = find(G(i,:) == 1); %Index of nodes where node i connects to below it
            Jleafs = intersect(Ii, Ileaf); %Index of leaves that node i eventually 
            %connects with
            nleafs = length(Jleafs); %number of leaves below Rcurr(i)
            misclass_Jleafs = 0;
            for j = 1:nleafs
                misclass_Jleafs = misclass_Jleafs + MisclassCost(Rpruned(Jleafs(j)), no_data_points); %Summations of misclassification of leaves
            end
            alpha(i) = ( MisclassCost(Rpruned(i), no_data_points) - misclass_Jleafs ) / ( nleafs - 1 ) ;
        end
    end
    [alpha.min, istar] = min(alpha);
    leaves_remove = find(G(istar,:) == 1);
    for leaf_no = 1:length(leaves_remove) %Remove Leaves
        Rpruned(leaves_remove(leaf_no)).I = NaN;
        Rpruned(leaves_remove(leaf_no)).p = NaN;
        Rpruned(leaves_remove(leaf_no)).j = NaN;
        Rpruned(leaves_remove(leaf_no)).s = NaN;
        Rpruned(leaves_remove(leaf_no)).left = NaN;
        Rpruned(leaves_remove(leaf_no)).right = NaN;
    end
    %Remove past information on the nodes, so it can be a leaf
    Rpruned(istar).left = NaN;
    Rpruned(istar).right = NaN;
    Rpruned(istar).j = NaN;
    Rpruned(istar).s = NaN;
    
    G(leaves_remove,:) = 0; %Replace the removed leaves with 0's in the Genealogy
    G(:,leaves_remove) = 0; %Replace the removed leaves with 0's in the Genealogy
    G(istar, :) = 0; %Node is now a leaf
    Ileaf = [Ileaf; istar]; %istar is now a leaf
    
    k = k + 1;
    T.R(k).k = Rpruned;
    T.R(k).a = alpha.min; %Store alpha
    T.R(k).G = G; %Store Genealogy Matrix
    
    %We create m number of trees some might be a repeat
end

%%%%%%%%%%%%%Test Pruned Trees on Test Data
for k = 1:m
    Test.R(k) = T.R(k);
    Test.R(k).k(1).I = 1:length(test_labels);
    
    %Tracking the leaves in the tree
    Ileafs = [];
    for h = 1:m
        %Current node metrics
        I = Test.R(k).k(h).I; %Pointer to the data points in the current rectangle
        j = Test.R(k).k(h).j;
        s = Test.R(k).k(h).s;
        p = Test.R(k).k(h).p;
        left = Test.R(k).k(h).left;
        right = Test.R(k).k(h).right;
        
        % Data in the current rectangle
        if (~isnan(I(1)))
            test_ind = test_data(:, I); 
        end
        
        if ((~isnan(j)) && (~isnan(s)) && (p(1)~=1)) %Only if node is not a leaf or empty node
            I_left = I(find(test_ind(j, :)<s));
            I_right = setdiff(I, I_left);
            
            %Compute the class distribution of the data points
            %Assign Left and Right Nodes data labels
            %Assign Left and Right Nodes frequency distributions
            if (~isempty(I_left))
               [p_left, ~, ~] = ClassDistr(test_labels, I_left);
               Test.R(k).k(left).I = I_left;
               Test.R(k).k(left).p = p_left;
            else %If empty, then nodes bottom of current node are empty
               Test.R(k).k(left).I = NaN;
               Test.R(k).k(left).p = NaN;
               Test.R(k).k(left).j = NaN;
               Test.R(k).k(left).s = NaN;
            end
            if (~isempty(I_right))
                [p_right, ~, ~] = ClassDistr(test_labels, I_right);
                Test.R(k).k(right).I = I_right;
                Test.R(k).k(right).p = p_right;
            else %If empty, then nodes bottom of current node are empty
               Test.R(k).k(right).I = NaN;
               Test.R(k).k(right).p = NaN;
               Test.R(k).k(right).j = NaN;
               Test.R(k).k(right).s = NaN;
            end
        elseif ((~isnan(j)) && (~isnan(s)) && (p(1)==1)) %For nodes that only have one data class,
            %but still have a split, from developing under the training data
            dead_leaves = [left, right]; %Identifying these leaves
            
            %Removing
            Test.R(k).k(h).j = NaN;
            Test.R(k).k(h).s = NaN;
            
            for y = 1:length(dead_leaves)
                Test.R(k).k(dead_leaves(y)).I = NaN;
                Test.R(k).k(dead_leaves(y)).p = NaN;
                Test.R(k).k(dead_leaves(y)).j = NaN;
                Test.R(k).k(dead_leaves(y)).s = NaN;
            end
            
            Ileafs = [Ileafs; h]; %Adds this h value into the leaf list
            %Because they become a leaf
        elseif ((isnan(j)) && (isnan(s)) && (~isnan(left) || ~isnan(right)))
            dead_leaves = [left, right]; %Identifying these leaves
            for y = 1:length(dead_leaves)
                Test.R(k).k(dead_leaves(y)).I = NaN;
                Test.R(k).k(dead_leaves(y)).p = NaN;
                Test.R(k).k(dead_leaves(y)).j = NaN;
                Test.R(k).k(dead_leaves(y)).s = NaN;
            end
        elseif (~isnan(I(1)) && ((isnan(j)) || (isnan(s)))) %Check if it is an empty node or leaf; empty node = do nothing
                Ileafs = [Ileafs; h];
        end
        
    end
  
    Ileaf_stored{k} = Ileafs; %Store the leaves of each tree in a cell array
end

misclass_test = zeros(1,m);
for k = 1:m
    Ileafs_temp = Ileaf_stored{k};
    for leaves_num = 1:length(Ileafs_temp)
      misclass_test(k) = misclass_test(k) + MisclassCost(Test.R(k).k(Ileafs_temp(leaves_num)), length(test_labels)); %Summations of misclassification of leaves
    end
end

%Find the optimal tree number
[~, optimal_tree_number] = min(misclass_test);

%Leaves of the optimal tree number
Optimal_leaves = Ileaf_stored{optimal_tree_number};

%Majority vote classications in the leaves
new_labels_vector = [];
new_labels_order = [];
for l = 1:length(Optimal_leaves)
    leaves_num = Optimal_leaves(l);
    I_leaves = Test.R(optimal_tree_number).k(leaves_num).I; %Data values from optimal tree
    [~,c,~] = ClassDistr(test_labels,I_leaves);
    new_labels_vector = [new_labels_vector, repmat(c,1,length(I_leaves))];
    new_labels_order = [new_labels_order, I_leaves];
end

[~,sorted_labels] = sort(new_labels_order);
new_labels = new_labels_vector(sorted_labels);

data_classified = [test_data(:,sorted_labels); new_labels];

end

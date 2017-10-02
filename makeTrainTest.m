function [train_label, test_label] = makeTrainTest(labels, split_train)
check_train = 0;
check_test = 0;
while(check_train==0 | check_test==0)
    %Randomly Initialize: Give a partitioning 
    labels_rand(1,:) = randperm(size(labels,2),size(labels,2)); %randi(k,size(labels,2),1)' is a row of randomly generated integers from 1 to k
    bins = round(size(labels,2) * split_train); %Set bins based off of size of k
    random = 1:2; %For train and test
    labels_rand(2,:) = 0; %Initialize labels 2nd row (our assignment of k labels)
    for i = 1:2
        labels_rand(2,(bins * (i - 1) + 1):size(labels,2)) = random(i);
    end
    if size(labels,2) < (bins * i)
        labels_rand = labels_rand(:,1:size(labels,2)); %Clean off if size of labels generated are higher than the size of the labels
    end
    %Sort the matrix but keep the column structure
    [~, order] = sort(labels_rand(1,:));
    labels_rand = labels_rand(:,order);
    label = labels_rand(2,:);
    %Make train and test label
    train_label = find(label == 1);
    test_label = find(label == 2);
    %Make train and test
    train = labels(:,train_label);
    test = labels(:,test_label);
    %Tabulate Train Test
    tab_train = tabulate(train);
    tab_test = tabulate(test);
    %Check for all classes are represented in train and test; if the counts
    %have zeroes or not all classes are shown in the tabulation, then there is
    %a violation
    check_train = (isempty(find(tab_train(:,2) == 0))) & (size(tab_train, 1)==length(unique(labels)));
    check_test = (isempty(find(tab_test(:,2) == 0))) & (size(tab_test, 1)==length(unique(labels)));
end

end
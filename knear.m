%K nearest neighbors classifier
function [new_annotations] = knear(train, annotations, test, k, p)
    for i = 1:size(train,2)
        for j = 1:size(test,2)
            D(i, j) = norm(train(:,i) - test(:,j), p); %rows are training data, columns are test data
        end
    end
    new_annotations = []; %Initialize New Annotations
    %for each given data vector
    for j = 1:size(test, 2)
        if ~isequal(train, test)
            %Find the k nearest data vectors x(j1), x(j2), ..., x(jk)
            [~, order] = sort(D(:,j), 'ascend'); %Find nearest train data vectors near test data vectors
            table = tabulate(annotations(order(1:k))); %Summary Statistics on Categories Frequencies
            [~, order_table] = sort(table(:,2), 'descend'); %Sort Summary Statistics, so top category shows up
            new_annotations(1,j) = table(order_table(1), 1); %Assign new annotations based on majority voting
        end
        if isequal(train, test) %In the case that the training data is testing data
            %Find the k nearest data vectors x(j1), x(j2), ..., x(jk)
            [~, order] = sort(D(:,j), 'ascend'); %Find nearest train data vectors near test data vectors
            table = tabulate(annotations(order(2:k+1))); %Summary Statistics on Categories Frequencies; top minimum value is always same data vector
            [~, order_table] = sort(table(:,2), 'descend'); %Sort Summary Statistics, so top category shows up
            new_annotations(1,j) = table(order_table(1), 1); %Assign new annotations based on majority voting
        end
    end
end
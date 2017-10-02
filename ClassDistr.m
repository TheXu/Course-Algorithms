function [p, c, r] = ClassDistr(C, I)
%Function that returns the relative frequencies of the different classes
%for the current rectangle, given the annotation vector C and an index
%vector listing the data points in the current rectangle.

%r represents the misclassification error
%c represents the majority vote

%Tabulate the classes
tabulate_C = tabulate(C(:,I));
[p, order] = sort(tabulate_C(:,3), 'descend');
p = p / 100; %Relative frequencies of different classes
tabulate_C = tabulate_C(order, :);
c = tabulate_C(1,1); %Majority vote

%Misclassification Error using Gini Index
r = 1 - sum(p .^ 2);


end
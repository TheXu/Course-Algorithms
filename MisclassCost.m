function misclass = MisclassCost(R, no_data_points)
%Computes the misclassification cost of a given node

%R is one of the nodes, Rcurr(j)

%Assuming equal cost in misclassification
r = 1 - R.p(1); %No cost if we get classification right, equal cost if we get classification wrong
v = length(R.I) / no_data_points;

misclass = r * v;
end

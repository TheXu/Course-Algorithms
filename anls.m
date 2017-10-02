function [final_W final_H t] = anls(data, k, tau)
%Intitialize W = W(0), wij>=0, t= 0
t = 0;
W = rand(size(data,1),k);
%W_t = ones(size(data,1), k)/(sqrt(size(data,1)*k)); %Initialize W(0)
W_t = W; %Initialize W(0)
W_t1 = W_t + ones(size(data,1),k) * (tau*999999); 
H_t = rand(k, size(data,2)); %Initialize H(0)
H_t1 = H_t;
while ( ( norm( (W_t1 - W_t) , 'fro')/norm(W_t, 'fro') ) + ( norm( (H_t1 - H_t) , 'fro')/norm(H_t, 'fro') ) ) > tau
    %While Loop Start
    W_t = W_t1;
    H_t = H_t1;
    H = []; %Initialize H
    for ell = 1:size(data, 2) %Iterate over columns up to p
        H = [H lsqnonneg(W, data(:,ell))]; %Update H(t + 1) = argmin||X-W(t)H||f subject to hij >= 0
    end
    W = []; %Re-Initialize W
    for j = 1:size(data, 1) %Iterate over rows up to n
        W = [W ; (lsqnonneg(H', data(j,:)'))']; %Update W(t + 1) = argmin||X-WH(t+1)||f subject to wij>=0
    end
    %Now scale the matrices
    lambdas = [];
    for i = 1:k
        lambdas = [lambdas norm(W(:,i), inf)]; %Calculate the max value of each column of W, 1 to k
    end
    L = diag(lambdas); %Used to scale the matrix for uniqueness
    W_t1 = W / L; %Scale the W. Also we fully define W(t+1)
    H_t1 = L * H; %Scale the H. Also we fully define H)t+1)
    t = t + 1; %Next iteration
end
final_W = W_t1;
final_H = H_t1;
end
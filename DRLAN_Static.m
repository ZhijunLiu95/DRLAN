function [U,U_list,U_S, Net_R] = DRLAN_static(Net, Attri, d, beta, alpha, p, q, theta, s)

%          Net   is the weighted adjacency matrix
%         Attri  is the attribute information matrix in a cell
%          d     is the dimension of the embedding representation
%         beta   is the weight for node structure information
%        alpha   is the weights for different order structure information
%        theta   is the weights for different order structure information
%          s     is the sparsity of random projection matrix
%          p     is the order of structure similarity
%          q     is the order of attribute similarity
%% Random Projection
start_time = tic;
N = length(Net);
[Net_R,root] = sparse_randmatrix(N,d,s);

% calculate structure embedding matrix U_S

U_list = cell(p+1,1);
U_list(1) = {Net_R};
for i = (2:p+1)
    U_list{i,1} = Net * sparse(U_list{i-1});
end
U_S = sparse(N,d);

% combine
for i = (1:p+1)
    U_S = U_S + alpha(i)*sparse(U_list{i});
end

% calculate attribute representation matrix U_A
[Attri_size,~] = size(Attri);
U_A_cell = cell(Attri_size,1);
ATR = cell(Attri_size,1);
pre_A = cell(Attri_size,1);
for i = (1:Attri_size)
    A_T = cell2mat(Attri(i))';
    temp = sum(A_T.^2).^.5;
    temp(find(temp==0))=0.0001;
    A_T = bsxfun(@rdivide, A_T, temp); % Normalize
    ATR{i,1} = sparse(A_T) * Net_R;
    pre_A{i,1} = A_T';
    U_A_t = A_T'* ATR{i,1}; % pre-projection
    U_A_list = cell(q+1,1);
    U_A_list{1,1} = Net_R;
    U_A_list{2,1} = U_A_t;
    for m = (3:q+1)
        U_A_t = A_T * U_A_t;
        U_A_t = A_T' * U_A_t;
        U_A_list{m,1} = U_A_t;
    end
    U_A_cell{i,1} = sparse(N,d);
    
    for j = (1:q+1)
        U_A_cell{i, 1} = U_A_cell{i,1} + theta(j)*sparse(U_A_list{j,1});
    end
end
clear A_T & S_A_P & U_A_t;

U_A = sparse(N,d);
for i = (1:Attri_size)
    U_A = U_A + U_A_cell{i,1};
end

% Obtain mixture embedding
U = beta*(U_S)+(1-beta)*(U_A);
U = U*root;
end_time = toc(start_time);
fprintf('Random Projection is end. time:%d minutes and %f seconds\n',floor(end_time/60),rem(end_time,60));
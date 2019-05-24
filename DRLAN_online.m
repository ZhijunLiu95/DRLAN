function [U_updated, U_list_updated, U_S_updated] = DRLAN_online(Net, Net_t, A_t, Y_list, U_S, beta, alpha, p, q, d, hyper, s)

%         Net    is the weighted adjacency matrix
%        Net_t   is the weighted adjacency matrix at new time step t
%         A_t    is the attribute matrix at new time step t
%       Y_list   is the previous structure projection results Y_0, Y_1, ...,Y_q
%         U_S    is the previous sturcture projection result
%          d     is the dimension of the embedding representation
%        beta    is the weight for node structure information
%        alpha   is the weights for different order structure information
%        theta   is the weights for different order structure information
%          s     is the sparsity of random projection matrix
%          p     is the order of structure similarity
%          q     is the order of attribute similarity
%% Data Preprocessing
% Match the dimension
[new_row,~] = size(Net_t);
[old_row,~] = size(Net);
D_row = new_row-old_row;
if D_row>0
    Net = [Net(:,:),sparse(old_row,D_row)]; % add row
    Net = [Net(:,:);sparse(D_row,new_row)]; % add col
end
[Attrisize,~]=size(A_t); % 有多少个属性矩阵
Net_d = Net_t - Net; % delta_A
%% Update Structure Embeddings

start_time = tic;
if D_row>0
    % match the dimension for U_list
    U_0_d = sparse_randmatrix(D_row, d, s); % R^hat
    zeromat = zeros(D_row, d);
    U_S = [U_S(:,:);zeromat];
    Y_list{1} = [Y_list{1};U_0_d];
    for i = 2:p+1
        Y_list{i} = [Y_list{i};zeromat];
    end
end
% update
D_U_list = cell(p+1,1);
D_U_list{1} = sparse(new_row,d);
U_list_updated = cell(p+1,1);
U_list_updated{1} = Y_list{1};
for i = (2:p+1)
    D_U_list{i,1} = Net_t * D_U_list{i-1,1} + Net_d * Y_list{i-1};
    U_list_updated{i,1} = Y_list{i,1} + D_U_list{i,1};
end
% combine
D_U_S = sparse(new_row,d);
for i = (1:p+1)
    D_U_S = D_U_S + alpha(i)*sparse(D_U_list{i});
end

U_S_updated = U_S + D_U_S;

%% Calculate Attribute Embeddings

U_A_cell = cell(Attrisize,1);
ATR = cell(Attrisize,1);
for i = 1:Attrisize
    At_T = A_t{i}';
    temp = sum(At_T.^2).^.5;
    temp(find(temp==0))=0.0001;
    At_T = bsxfun(@rdivide, At_T, temp); % Normalize
    ATR{i,1} = sparse(At_T) * Y_list{1};
    U_A_t = At_T'* ATR{i,1}; % projection
    U_A_list = cell(q+1,1);
    U_A_list{1,1} = Y_list{1};
    U_A_list{2,1} = U_A_t;
    for m = (3:q+1)
        U_A_t = At_T * U_A_t;
        U_A_t = At_T' * U_A_t;
        U_A_list{m,1} = U_A_t;
    end
    U_A_cell{i,1} = sparse(new_row,d);
    
    for j = (1:q+1)
        U_A_cell{i, 1} = U_A_cell{i,1} + hyper(j)*sparse(U_A_list{j,1});
    end
end
U_A_updated = sparse(new_row,d);
for i = (1:Attrisize)
    U_A_updated = U_A_updated + U_A_cell{i,1};
end
%% updating embeddings
U_updated = beta*(U_S_updated)+(1-beta)*(U_A_updated);
end_time = toc(start_time);
fprintf('Update Random Projection is end. time:%d minutes and %f seconds\n',floor(end_time/60),rem(end_time,60));
end


% Code for Dynamic Representation Learning for Large-scale Attributed Network

Dataset = 'Flickr';

t_start = tic;
if strcmp(Dataset,'BlogCatalog')
    load('data\BlogCatalog\BlogCatalog_times.mat'); % time stamped dataset
    % Parameters for Network Reconstruction
    p = 2; % order of structure similarity
    q = 1; % order of attribute similarity
    beta = 0.95; % weight of network structure information for embedding
    alpha = [1,0.1,0.001,0.01,0.001]; % weights for different order structure information
    theta = [1,1,0.01,1,1]; % weights for different order attribute information
    
    % Parameters for Node Classification
    %             p=4;
    %             q=3;
    %             beta = 0.88;
    %             alpha = [10,0.01,0.01,0.01,0.001];
    %             theta = [100,10,10,1];
    
    s =1; % sparsity of random projection matrix
elseif strcmp(Dataset,'Flickr')
    load('data\Flickr\Flickr_times.mat')
    % Parameters for Network Reconstruction
    p = 2;
    q = 1;
    beta = 0.92;
    alpha = [1,0.1,0.001,0.0001];
    theta = [1,1,0.1,0.001,0.0001];
    % Parameters for Node Classification
    %             p=4;
    %             q=3;
    %             beta = 0.92;
    %             alpha = [1, 0.1, 0.1, 0.01, 0.01, 0.0001];
    %             hyper = [1, 1, 1000,1000,1000,1000];
    s = 20;
elseif strcmp(Dataset,'DBLP')
    load('data\DBLP\DBLP_times.mat');
    % Parameters for Network Reconstruction
    p=2;
    q=1;
    beta =0.9;
    alpha = [1,0.1,0.001];
    theta = [1,0.001,0.01];
    
    % Parameters for Node Classification
%     p = 2;
%     q = 2;
%     beta = 0.95;
%     alpha = [1,1,0.1,0.1,1];
%     theta = [1,10,100];
    s = 1;
end

d = 128; % the dimension of the embedding representation

for j = 1:length(Network)
    %% data processing
    if j==1
        A = cell2mat(Network(j));
        if strcmp(Dataset,'DBLP')
            X = Attributes{j};
        else
            X = Attributes(j);
        end
        label = cell2mat(Label(j));
        [Net_row,~] = size(A);
        Net = A;
    end
    
    if j~=1
        A_t = Network{j};
        if strcmp(Dataset,'DBLP')
            X_t = Attributes{j};
        else
            X_t = Attributes(j);
        end
        label = Label{j};
        Net = A_t;
    end
    
    %% Large Scale Attributed Network Embedding
    if j==1
        disp('Unsupervised Attributed Network Embedding (Random Projection):')
        [U,U_list,U_S, R] = DRLAN_Static(A,X,d,beta,alpha,p, q,theta,s);
    end
    if j~=1
        [U, U_list_updated, U_S_updated] = DRLAN_online(A, A_t, X_t, U_list, U_S, beta, alpha, p, q, d, theta, s);
    end
    
    filename = "data\";
    filename = strcat(filename,Dataset,"\embedding_",num2str(j),".mat");
    U = full(U);
    save(filename, 'U');
    
    if j~=1  % for update
        A=A_t;
        U_list = U_list_updated;
        U_S = U_S_updated;
    end
end


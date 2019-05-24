%This function return a matrix R which generated according to 
%the very sparse matrix distrubution defined in the paper 
%"Very Sparse Random Projections" KDD'06

function [R, s_root]=sparse_randmatrix(cols,d,fold)
    %Generate the very sparse random matrix
    R=zeros(cols,d);
    s = sqrt(cols)*fold;
    s_root = sqrt(s);
    for i=1:cols
        for j=1:d
            temp=rand;
            if(temp<1/(2*s))%if temp lies in the range [0,1/(2sqrt(cols))], i.e. w.p. 1/6, R(i,j)=sqrt(3)
                R(i,j)=1;
            elseif(temp>(1-1/(2*s)))%if temp lies in the range [1-(1/(2sqrt(cols))),1], i.e. w.p. 1/6, R(i,j)=-sqrt(3)
                R(i,j)=-1;
            else%if temp lies in the range [1/(2sqrt(cols)),1-(1/(2sqrt(cols)))], i.e. w.p. 2/3, R(i,j)=0
                R(i,j)=0;
            end      
        end
    end
    R = sparse(R);
end
function [Z,A,H,G,E,S,converge] = ASCR(X, gt, index, k, c, param)

alpha = param.alpha; 
%% Normalized data
num_view = length(X);%number of views/features
n = numel(gt);% number of samples
dk=[]; POS1 = []; POS2=[];
for i=1:num_view
    X{i} = NormalizeData(X{i});
%     POS1{i} =  find(index(:,i)==1);   %existing sample position
    POS2{i} =  find(index(:,i)==0);   %missing sample position
end
Xc = X;

%% initialization
for i = 1:num_view   
    di = size(X{i},1);
    A{i} = zeros(di,k);
    E{i} = zeros(n,k);
    H{i} = zeros(n,c);
    G{i} = zeros(k,c);
end


flag = 1;
iter = 0;
%%
MAX_iter=30;
while flag
    iter = iter + 1;
    for i=1:num_view
     Hpre{i}=H{i};
    end
     if mod(iter, 10)==0
      fprintf('%d..',iter);
     end
     
    %E step
    S=[]; 
    for i=1:num_view
        S{i}=H{i}*G{i}'-X{i}'*A{i};
        E{i}(POS2{i},:) = S{i}(POS2{i},:);
    end  
    
    % A step
    for i=1:num_view
        [U,~,V] = svd(Xc{i}*(H{i}*G{i}'-E{i}),'econ');
        A{i} = U*V';
        clear  U V;
    end  
    
    % G step
    for i=1:num_view
        [U,~,V] = svd((Xc{i}'*A{i}+E{i})'*H{i},'econ');
        G{i} = U*V';
        clear  U V;
    end  
    
    % H step
    for i=1:num_view
        OO{i} = (Xc{i}'*A{i}+E{i})*G{i};
    end 
    OO_tensor = cat(3, OO{ : , : });
    OOv = OO_tensor(:);
    [Lv, ~] = wshrinkObj(OOv, alpha/2, [n, c, num_view], 0, 3);
    H_tensor = reshape(Lv, [n, c, num_view]);
    for i=1:num_view
        H{i} = H_tensor(:,:,i);
    end 

    term1 = 0;
    term2 = 0;

    for i = 1:num_view
        term1 = term1 + norm(H{i}-Hpre{i},'fro')^ 2;
        term2 = term2 + norm(Hpre{i},'fro')^2;
    end
        converge(iter)=term1/term2;
    if (iter>1) && (converge(iter)<1e-5 || iter>MAX_iter )
         flag = 0;
    end
    
end

Z=0;
Z=cell2mat(H);
end

function [Wx,Wy,r] = cluster_cca(train_a,train_b,a_labels,b_labels,kapa_cca)
%keyboard;
mean_a = mean(train_a,2);
mean_b = mean(train_b,2);

%train_a = train_a - repmat(mean_a,[1,size(train_a,2)]);
%train_b = train_b - repmat(mean_b,[1,size(train_b,2)]);

%% Get the center of each clusters in both the sets (MEAN CCA)
% disp('>Getting the center of each clusters....');
unq_a_label = unique(a_labels); %1x100
unq_b_label = unique(b_labels); %1x100

%% Calculating the cardinality of all the classes in both the sets
% disp('>Calculating the cardinality of all the classes in both the sets...');
card_a = zeros(1,size(unq_a_label,2)); %1x100
card_b = zeros(1,size(unq_b_label,2)); %1x100
for i=1:size(card_a,2)
    c = 0; d = 0;
    for j=1:size(a_labels,2)
        if unq_a_label(i)==a_labels(j)
            c = c + 1;
        end
    end
    for j=1:size(b_labels,2)
        if unq_b_label(i)==b_labels(j)
            d = d + 1;
        end
    end    
    card_a(1,i) = c;
    card_b(1,i) = d;
end

%% Calculate the value of the constant M
% disp('>calculating the value of M...');
M = 0;
for i=1:size(unq_a_label,2)
    M = M + card_a(1,i)*card_b(1,i);
end

%% Calculating the covariance matrix Cxx
% n = number of classes
% card = cardinality of the classes in each set
% disp('...calculating the covariance matrix Cxx....')
C = size(unq_a_label,2);
Cxx = 0;
for c=1:C %for each class
    %find those vectors having that label
    [~,idx]=find(a_labels==unq_a_label(c));
    sum = 0;
    for j=1:length(idx)
        x = train_a(:,idx(j));
        sum = sum + x*x.';
    end
    sum = card_b(1,c)*sum;
    Cxx = Cxx + sum;
end
Cxx = Cxx./M;
Cxx = Cxx + kapa_cca*eye(size(train_a,1));
 
%% Calculating the covariance matrix Cyy
% disp('...calculating the covariance matrix Cyy....')
C = size(unq_a_label,2);
Cyy = 0;
for c=1:C %for each class
    %find those vectors having that label
    [~,idx]=find(b_labels==unq_b_label(c));
    sum = 0;
    for j=1:length(idx)
        y = train_b(:,idx(j));
        sum = sum + y*y.';
    end
    sum = card_a(1,c)*sum;
    Cyy = Cyy + sum;
end
Cyy = Cyy./M + kapa_cca*eye(size(train_b,1));

%% Calculating the covariance matrix Cxy
% disp('...calculating the covariance matrix Cxy....')
% % 1st method
% C = size(unq_a_label,2);
% Cxy = 0;
% for c=1:C %for each class
%     %find those vectors in set a having that label
%     [~,idx1]=find(a_labels==unq_a_label(c));
%     %find those vectors in set a having that label
%     [~,idx2]=find(b_labels==unq_b_label(c));
%     for j=1:length(idx1)
%         x = train_a(:,idx1(j));
%         for k=1:length(idx2)
%             y = train_b(:,idx2(k));
%             Cxy = Cxy + x*y.';
%         end
%     end
% end
% Cxy = Cxy./M;
% Cyx = Cxy.';

%% 2nd method
% Reordering the matrix
train_a = train_a.';
train_b = train_b.';
% initialization the matrix
train_a_mean = zeros(length(unq_a_label),size(train_a,2));
train_b_mean = zeros(length(unq_b_label),size(train_b,2));
% calculating the means of the clusters
for i=1:length(unq_a_label)
    sum = 0; count = 0;
    [~,idx]=find(a_labels==unq_a_label(i));
    for j=1:length(idx)        
        sum = sum + train_a(idx(j),:);
        count = count + 1;
    end
    train_a_mean(i,:) = sum/count;
end
for i=1:length(unq_b_label)
    sum = 0; count = 0;
    [~,idx]=find(b_labels==unq_b_label(i));
    for j=1:length(idx)        
        sum = sum + train_b(idx(j),:);
        count = count + 1;
    end
    train_b_mean(i,:) = sum/count;
end

C = size(unq_a_label,2);
Cxy = 0;
for c=1:C %for each class
    mu_x = train_a_mean(c,:).';
    mu_y = train_b_mean(c,:).';
    Cxy = Cxy + card_a(1,c)*card_b(1,c)*mu_x*mu_y.';
end
Cxy = Cxy./M;
Cyx = Cxy.';

% Reordering the matrix
train_a = train_a.';
train_b = train_b.';

% disp('...calculating the projection matrices....')
%% Calculating the Wx cca matrix
Rx = chol(Cxx);
inv_Rx = inv(Rx);
Z = inv_Rx'*Cxy*(Cyy\Cyx)*inv_Rx;
Z = 0.5*(Z' + Z);  % making sure that Z is a symmetric matrix
[Wx,r] = eig(Z);   % basis in h (X)
r = sqrt(real(r)); % as the original r we get is lamda^2
Wx = inv_Rx * Wx;   % actual Wx values

%% Calculating Wy
Wy = (Cyy\Cyx) * Wx; 

% by dividing it by lamda
Wy = Wy./repmat(diag(r)',size(train_b,1),1);

return;
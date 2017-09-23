function [Wx, Wy, r] = mean_cca(X,Y,X_labels,Y_labels,k)

X = X.'; Y = Y.';
%% Get the center of each clusters in both the sets (MEAN CCA)
% disp('>Getting the center of each clusters....');
unq_X_label = unique(X_labels); %1x100
unq_Y_label = unique(Y_labels); %1x100
X_mean = zeros(length(unq_X_label),size(X,2)); %100x220
Y_mean = zeros(length(unq_Y_label),size(Y,2)); %100x220
% calculating the means of the clusters
for i=1:length(unq_X_label)
    sum = 0; count = 0;
    [~,idx]=find(X_labels==unq_X_label(i));
    for j=1:length(idx)        
        sum = sum + X(idx(j),:);
        count = count + 1;
    end
    X_mean(i,:) = sum/count;
end
for i=1:length(unq_Y_label)
    sum = 0; count = 0;
    [~,idx]=find(Y_labels==unq_Y_label(i));
    for j=1:length(idx)        
        sum = sum + Y(idx(j),:);
        count = count + 1;
    end
    Y_mean(i,:) = sum/count;
end
X = X_mean.'; Y = Y_mean.';

% calculating the covariance matrices
z = [X; Y];
C = cov(z.');
sx = size(X,1);
sy = size(Y,1);
Cxx = C(1:sx, 1:sx) + k*eye(sx);
Cxy = C(1:sx, sx+1:sx+sy);
Cyx = Cxy';
Cyy = C(sx+1:sx+sy,sx+1:sx+sy) + k*eye(sy);

%calculating the Wx cca matrix
Rx = chol(Cxx);
invRx = inv(Rx);
Z = invRx'*Cxy*(Cyy\Cyx)*invRx;
Z = 0.5*(Z' + Z);  % making sure that Z is a symmetric matrix
[Wx,r] = eig(Z);   % basis in h (X)
r = sqrt(real(r)); % as the original r we get is lamda^2
Wx = invRx * Wx;   % actual Wx values

% calculating Wy
Wy = (Cyy\Cyx) * Wx; 

% by dividing it by lamda
Wy = Wy./repmat(diag(r)',sy,1);
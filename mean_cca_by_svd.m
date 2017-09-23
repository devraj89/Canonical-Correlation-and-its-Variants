function [A,B,r,U,V] = mean_cca_by_svd(x,y,a_labels,b_labels)

unq_x_label = unique(a_labels);
unq_y_label = unique(b_labels);
x_mean = zeros(length(unq_x_label),size(x,2));
y_mean = zeros(length(unq_y_label),size(y,2));

%% Get the center of each clusters in both the sets (MEAN CCA)
disp('>Getting the center of each clusters....');
% calculating the means of the clusters
for i=1:length(unq_x_label)
    sum = 0; count = 0;
    [~,idx]=find(a_labels==unq_x_label(i));
    for j=1:length(idx)        
        sum = sum + x(idx(j),:);
        count = count + 1;
    end
    x_mean(i,:) = sum/count;
end
for i=1:length(unq_y_label)
    sum = 0; count = 0;
    [~,idx]=find(b_labels==unq_y_label(i));
    for j=1:length(idx)        
        sum = sum + y(idx(j),:);
        count = count + 1;
    end
    y_mean(i,:) = sum/count;
end

x = x_mean;
y = y_mean;

% computing the means
N = size(x,1);
% mu_x = mean(x,1); 
% mu_y = mean(y,1);
% % substracting the means
% x = x - repmat(mu_x,N,1); y = y - repmat(mu_y,N,1);

x = x.'; y = y.';
% computing the covariance matrices
Cxx = (1/N)*x*(x.');  Cyy = (1/N)*y*(y.'); Cxy = (1/N)*x*(y.');

%dimension
m = min(rank(x),rank(y));

% computing the quare root inverse of the matrix
[V,D]=eig(Cxx); d = diag(D);
% Making all the eigen values positive
d = (d+abs(d))/2; d2 = 1./sqrt(d); z =  d2(~isinf(d2));
d2(d==0)=max(z); Cxx_iv=V*diag(d2)*inv(V);

% computing the quare root inverse of the matrix
[V,D]=eig(Cyy); d = diag(D);
% Making all the eigen values positive
d = (d+abs(d))/2; d2 = 1./sqrt(d); z =  d2(~isinf(d2));
d2(d==0)=max(z); Cyy_iv=V*diag(d2)*inv(V);

Omega = Cxx_iv*Cxy*Cyy_iv;
[C,Sigma,D] = svd(Omega);
A = Cxx_iv*C; A = A(:,1:m);
B = Cyy_iv*D; B = B(:,1:m);
A = real(A); B = real(B);
U = A.'*x; V = B.'*y;
r = Sigma(1:m,1:m);
function [A,B,r,U,V] = cca_by_svd(x,y)

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

% computing the square root inverse of the matrix
[V,D]=eig(Cxx); d = diag(D);
% Making all the eigen values positive
d = (d+abs(d))/2; d2 = 1./sqrt(d); z =  d2(~isinf(d2));
d2(d==0)=max(z); Cxx_iv=V*diag(d2)*inv(V);

% computing the square root inverse of the matrix
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
This has the Matlab implementation of the non-kernel
versions of the algorithms 
(1) Canonical Correlation Analysis (CCA)
(2) Mean CCA
(3) Cluster CCA

The article can be found here 
http://proceedings.mlr.press/v33/rasiwasia14.pdf

I have done the implementations in two ways.
Kindly please feel free to use the one which suits you the most.

The code for the CCA implementation has been provided from
http://www.davidroihardoon.com/Professional/Code_files/cca.m

The rest of the implementations are mine

Kindly follow this syntax to use these codes
---------------------------------------------
Please note that it is assumed that you have already centered the data before feeding it into these functions.

[1] cca.m 

[Wx, Wy, r] = cca(X,Y,k)

Here k is the regularizer
X and Y are data matrices where each column is an observation and each row is an variable. X and Y is of size (dx X N) and (dy X N) resp. where dx and dy are the input data dimensions and N is the number of samples.

[2] cca_by_svd.m

[A,B,r,U,V] = cca_by_svd(x,y)

x and y are data items with size of (N X dx) and (N x dy)
A, B are the cca learned directions
r is the amount of correlation
U,V is the projected data x,y in the cca directions

[3] mean_cca.m

[Wx, Wy, r] = mean_cca(X,Y,X_labels,Y_labels,k)

k is the regularizer
X,Y are the data items of size (dx X N) and (dy X N) resp.
X_labels,Y_labels are the labels of the data.
Wx, Wy are the cca directions
r is the amount of correlation

[4] mean_cca_by_svd.m

[A,B,r,U,V] = mean_cca_by_svd(x,y,a_labels,b_labels)

x,y are the data items of size (N X dx) and (N X dy) resp.
a_labels, b_labels are the labels of data x,y
A, B are the cca learned directions
r is the amount of correlation
U,V is the projected data x,y in the cca directions


[5] cluster_cca.m

[Wx,Wy,r] = cluster_cca(train_a,train_b,a_labels,b_labels,kapa_cca)

kapa_cca is the regularizer
train_a,train_b are the data items of size (dx X N) and (dy X N) resp.
a_labels,b_labels are the labels of the data.
Wx, Wy are the cca directions
r is the amount of correlation


[6] cluster_cca_by_svd

[Wx,Wy,r] = cluster_cca_by_svd(train_a,train_b,a_labels,b_labels,kapa_cca)

kapa_cca is the regularizer
train_a, train_b are the data items of size (dx X N) and (dy X N) resp.
a_labels, b_labels are the labels of data train_a, train_b
Wx, Wy are the cca learned directions
r is the amount of correlation

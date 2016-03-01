function eucd_matrix = cveucdist(x, y)
eucd_matrix = [];
N = size(x,2);
P = size(y,2);
D = size(x,1);
% Inputs ([]s are optional)
%   (matrix) X        D x N matrix where D is the dimension of vectors
%                     and N is the number of vectors.
%   (matrix) [Y]      D x P matrix where D is the dimension of vectors
%                     and P is the number of vectors.
%                     If Y is not given, the L2 norm of X is computed and
%                     1 x N matrix (not N x 1) is returned.
%
% Outputs ([]s are optional)
%   (matrix) d        N x P matrix where d(n,p) represents the squared
%                     euclidean distance between X(:,n) and Y(:,p).
% 
% Examples
%   X = [1 2
%        1 2];
%   Y = [1 2 3
%        1 2 3];
%   d = cvEucdist(X, Y)
%       0     2     8
%       2     0     2
for n = 1 : N
    for p = 1 : P
        sum = 0;
        a = y(:,p);
        b = x(:,n);
        for d = 1 : D
            sum = sum + (a(d) - b(d))^2;
        end
        eucd_matrix(n,p) = sum;
    end
end

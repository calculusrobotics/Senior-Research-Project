format long;

A = [
    1, 1.1;
    0, 0.5
];
B = [0; 1];
Q = [
    1, 0.5;
    0.5, 2
];

x00 = [1; 0];
x = mvnrnd(x00', Q)';

H = [1, 0];
R = 0.6;

P00 = Q;

% propogate
P10 = A * P00 * A' + Q;
x10 = A * x00 + B * 2;
x = A * x + B * 2 + mvnrnd(zeros(1, 2), Q)';

% update
y = H*x + normrnd(0, R);
S = H * P10 * H' + R;
K = P10 * H' / S;
P11 = (eye(2) - K * H) * P10;

x11 = x10 + K*(y - H*x10);

% estimation set up
N = 100;
NV = 200;
boundX = 5*sqrt(P10(1));
boundV = 5*sqrt(R);

deltaX = 2*boundX / N;

xs = linspace(x10(1) - boundX, x10(1) + boundX, N);

v = -boundV + 2*boundV*rand([1, NV]);
%v = linspace(-boundV, boundV, NV);

cdf = @(v) normcdf(v, 0, sqrt(R));

cdfs = cdf(v);

Mat = zeros(NV, N);
for i = 1:NV
    for j = 1:N
        xj = [xs(j); 0];
        
        if y - H*xj <= v(i)
            Mat(i, j) = 1;
        end
    end
end

Mat = Mat * deltaX;

rhos = pinv(Mat) * cdfs';

%
boundVel = 5*sqrt(P10(4));
M = N;
vels = linspace(x10(2) - boundVel, x10(2) + boundVel, M);

deltaVel = boundVel * 2 / M;

MSE = zeros(2, 2);

for j = 1:N
    for l=1:M
        xjl = [xs(j); vels(l)];
        
        MSE = MSE + mvnpdf(xjl, x10, P10) * rhos(j) * (xjl - x11)*(xjl-x11)';
    end
end

MSE = MSE * deltaX * deltaVel;

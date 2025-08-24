clc; clear; close all;

% System matrices from Example 6.1
A = [1 -2; 1 4];
B = [1; 2];
N = [0.1; 0.5];
C = [1 0.5];
D = 1;
E = [1; 1];
H1 = [0.1 0.2];
H2 = 0.1;

% Dimensions
n = size(A,1);
m = size(B,2);
s = size(H1,1);
p = size(C,1);

% YALMIP decision variables
Pbar = sdpvar(n,n,'symmetric');
Kbar = sdpvar(m,n,'full');
epsilon = sdpvar(1);
gamma = sdpvar(1);

% Define expressions
Cbar = C + D*Kbar;
Abar = A*Pbar + B*Kbar;
HbarP = H1*Pbar + H2*Kbar;

% LMI formulation
LMI = [...
    Abar + Abar' + epsilon*(E*E'),      Cbar',              HbarP',         Pbar*N;
    Cbar,                              -eye(p),            zeros(p,s),     zeros(p,1);
    HbarP,                             zeros(s,p),         -epsilon*eye(s), zeros(s,1);
    N'*Pbar,                           zeros(1,p),         zeros(1,s),     -gamma^2 ];

% Constraints
constraints = [Pbar >= 1e-6*eye(n), epsilon >= 1e-6, gamma >= 1e-6, LMI <= -1e-6];

% Objective: minimize gamma
options = sdpsettings('solver','sedumi','verbose',1);
sol = optimize(constraints, gamma, options);


% Results
if sol.problem == 0
    gamma_val = value(gamma);
    Pbar_val = value(Pbar);
    Kbar_val = value(Kbar);
    K = Kbar_val / Pbar_val;

    fprintf('\n✅ Minimum gamma: %.4f\n', gamma_val);
    fprintf('✅ Control gain K:\n');
    disp(K);
else
    disp('❌ Optimization failed:');
    disp(sol.info);
end

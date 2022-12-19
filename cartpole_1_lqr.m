clc;
clear;

%Put into state space form
A = [0 1 0      0; ...
     0 0 0.7164  0;...
     0 0 0      1; ...
     0 0 15.76 0];
B = [0;0.9755;0;1.46];
C = eye(4);
D = 0;
sys = ss(A,B,C,D);

%Check controllability
if rank(ctrb(sys))==4
    disp('System is controllable!');
else
    disp('System is not controllable :(');
end

%LQR Controller
Q = [10 0 0 0; 0 1 0 0; 0 0 10 0; 0 0 0 1]; %cost on states
R = 10; %cost on inputs
[K,S,E] = lqr(sys,Q,R);

%Simulation Parameters
theta_initial = 45 ;
dtheta_initial = 0;
%Run Simulation
sim('cartpole_model_1.slx');






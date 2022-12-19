clc;
clear;

%Simulation Parameters
theta_initial = 70 ;
dtheta_initial = 0;
K=[-2 -0.4];
sim('cartpole_model_2.slx');


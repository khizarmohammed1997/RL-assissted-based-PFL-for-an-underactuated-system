
function u=feedback_linearization_simscape(theta_val,theta_dot_val,virtual_input)
    %syms mc mp theta x theta_dot x_dot l f g x_ddot theta_ddot
    syms theta theta_dot theta_ddot
    %syms theta_val theta_dot_val virtual_input
    
%     L=0.5;
%     g=9.8;
%     m_cart=1; The 
%     m_pole=1;
%     
%     eq1= (mc+mp)*x_ddot +mp*l*theta_ddot*cos(theta)-mp*l*theta_dot^2*sin(theta);
%     eq2=mp*l*x_ddot*cos(theta)+mp*l^2*theta_ddot-mp*g*l*sin(theta);
%     
%     x_ddot_sub=solve(eq2,x_ddot);
%     u=subs(eq1,x_ddot,x_ddot_sub);
%     
%     old_vals=[mc mp l g];
%     new_vals=[m_cart m_pole L g];
%     u=subs(u,old_vals,new_vals)
    
    u_equation=(theta_ddot*cos(theta))/2 - (theta_dot^2*sin(theta))/2 - (2*((5*theta_ddot)/2 - 49*sin(theta)))/(5*cos(theta));
    u=double(subs(u_equation,[theta, theta_dot, theta_ddot],[theta_val,theta_dot_val,virtual_input]));
---
layout: page
title: Robotics
permalink: /robotics/
---

<style>
.fa-file-code-o {
    font-size: 18px; /* Adjust size as needed */
}

.fa-file-pdf-o {
	font-size: 18px; /* Adjust size as needed */
}


</style>


<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PQFC01D0LX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PQFC01D0LX');
</script>

<!-- TODO: add more projects:
- vision based robot localization
- contact force model
 -->

## Robotics & Control Projects
- [Panda for Hyperthermia](#panda-for-hyperthermia)
- [MPC for Soft Robots](#mpc-for-soft-robots)
- [Motion Planning with RRT + CBF](#motion-planning-with-rrt--cbf)
- [Aircraft Landing Control](#aircraft-landing-control)
- [Trajectory Tracking of a KUKA LBR 7R](#trajectory-tracking-of-a-kuka-lbr-7r)



## Panda for Hyperthermia <span id="panda-for-hyperthermia" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://github.com/Emanuele-n/panda-win)</span>
While working at [Medlogix](https://www.albahyperthermia.com/) I had the possibility of writing the code for controlling the [Panda](https://robodk.com/robot/Franka/Emika-Panda) robot from Franka Emika. Here I experimented with many different control techniques such as FBL, impedance and admittance control, to find the best solution for curing cancer with hyperthermia! <br/>
One day I asked chatGPT to write the parametric Cartesian equations of an heart and I made the robot follow the trajectory, basically I am just a bridge between two robots. <br/> 
Here is the Panda spreading some love
<p align="center">
	<img src="/media/love_trajectory.gif"/>
</p>

### Short Summary 
**Goal**: Control the Panda robot for delivering hyperthermia treatment ensuring safety and compliance with the patient body
- Dynamic **model** of a robot in contact
<p align="center">
<span>$$ M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = u + J^T_r (q)F_r $$</span><br>
</p>
- Impedance **control** law for constant reference $$ r_d $$ <br/> By choosing the desired (apparent) inertia equal to the natural Cartesian inertia of the robot we don't need force feedback, and we can impose the desired damping $$ D_m $$ and stiffness $$ K_m $$ to the robot!
<p align="center">
<span>$$ u = J^T_r (q)[K_m (r_d - r) -D_m \dot{r}] + G(q) $$</span><br>
</p>

<small>Last Update: January, 2023</small>  



## MPC for Soft Robots <span  id="mpc-for-soft-robots" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://www.mathworks.com/matlabcentral/fileexchange/104060-soropcc)</span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://drive.google.com/file/d/1wzDfhW_K8pfrNxatST2SaEYtEBRyG7u8/view?usp=share_link)</span>


While I was writing my Master's [thesis](https://drive.google.com/file/d/1wzDfhW_K8pfrNxatST2SaEYtEBRyG7u8/view?usp=share_link) I decided to share the code I was using as a new MATLAB [Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/104060-soropcc). SoRoPCC  allows you to simulate and control a customizable soft manipulator, both for Shape Control and for Task-space Control, either by controlling the curvature of each CC segment or the position and orientation of the tip of the robot. Moreover it is also embedded with the MPC Toolbox, so you can control the robot using the Model Predictive Control.<br/>
Here are some examples of trajectory tracking and set-point regulation of a 3-DOF tentacle.

<p align="center">
	<img src="/media/track_fully.gif" width="350" height="300"/>
	<img src="/media/track_under.gif" width="350" height="300"/>
</p>

<p align="center">
	<img src="/media/track_plus_alpha.gif" width="350" height="300"/>
	<img src="/media/reg_obstacle.gif" width="350" height="300"/>
</p>

Top-left is fully actuated; top-right is underactuated on the last CC segment; bottom-left shows how it is possible to control also the orientation; bottom-right is an attempt of sliding on an obstacle to exploit the compliance structure of the robot.

### Short Summary
**Goal**: Executing tasks with a Soft Continuum Robot
- Dynamic **model** of a soft robot with $$ n $$ constant curvature segments $$ q \in \mathbb{R}^n $$
<p align="center">
  <span>$$ M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) + D(q)\dot{q} + Kq = A\tau $$</span><br>
</p>
- Discrete time state space representation 
<p align="center">
  <span>$$ x_{k+1} = f(x_k , u_k ) $$  $$  y_k = h(x_k) $$</span><br>
</p>
- MPC **control** law is obtained by minimizing the cost function (with tail cost) and ensuring the state and input constraints
<p align="center">
  <span>$$ J_{M}^{N}(x,u) = \sum_{i = 0}^{M-1}l(x(k),u(k)) + \sum_{i = M}^{N-1}l_{\tau}(x(k))  $$</span><br>
</p>
- What about the closed loop **stability**? <br/> 
	It has been proved that a sufficiently large prediction horizon guarantees the stability of the system. I was very curious of finding the numerical value of the lower bound $$ \bar{N} $$. <br/>
	I learned that if a known stabilizing control law is used to compute the tail cost, this one is a relaxed control Lyapunov function which implies the asymptotic stability and many estimates of the lower bound have been proposed. I used the one proposed by [J. Köhler](https://doi.org/10.1016/j.ifacol.2021.08.540) and as stabilizing control laws I used some of those proposed by my friend and colleague [P. Pustina](hthttps://scholar.google.com/citations?user=IiaVyHAAAAAJ&hl=en)
	
- Playing with the cost index to use the soft robot compliance <br/>
	The general form that I used for the stage cost is the quadratic norm with respect to a positive definite matrix, both for the state and the input, which is a common choice for the MPC. 
	<p align="center">
	<span>$$ l(x,u) = ||x(k)||^2_Q + ||u(k)||^2_Q $$</span><br>
	</p>
	However, I wanted to try to reduce the rigidy of the robot, which is a common issue when applying standard control techniques to soft robots. So I tried to replace the constant weight multiplying the control effort with a row vector  $$ s = [s_1 ,  s_2,   \cdot \cdot \cdot , s_m]^{T} $$ of the same dimension of the control vector, where $$ s_i $$ is the $$i$$-th input weight, yelding to the cost function
	<p align="center">
	<span>$$\begin{array}{c}
		J_{ts}(x,u) = 
		\sum_{k = 0}^{M-1}
			a_1 \norm{h(x(k)) - y_d (k)}^{2} + 
			a_2 \norm{\dot{q}(k)}^{2} +
			a_3 \norm{s \cdot u(k)}^{2} + \\
			a_4 \norm{u(k+1)-u(k)}^{2} + 
			\sum_{k = M}^{N-1} 
			a_5\norm{h(x(k)) -  y_d (k)}^{2} + 
			a_6 \norm{\dot{q}(k)}^{2}
		\end{array}$$</span><br>
	</p>
	Note that the latters by regulating $$ s $$ one can "decide" the severity of the underactuation of the system according to the magnitude of the weight. The idea was to dynamically adjust the weights to make the robot more compliant when needed, for example when it is in contact with an obstacle, exploiting the compliance of the structure (see simulation above)
	

<small>Last Update: March, 2024</small>  


## Motion Planning with RRT + CBF <span id="motion-planning-with-rrt--cbf" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://github.com/Emanuele-n/Enhancing-kinodynamic-RRT-using-CBF-based-steering) </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://github.com/Emanuele-n/Enhancing-kinodynamic-RRT-using-CBF-based-steering/blob/main/AMR_report.pdf)</span>
Here is a project about Probabilistic Motion Planning of Unicyle Mobile Robot using Rapidly-exploring Random Tree (RRT) with Control Barrier Functions (CBF)

Here are some examples of the application of the algorithm in four scenes with increasing difficulty
<p align="center">
	<img src="/media/scene1.gif" width="300" height="200"/>
	<img src="/media/scene2.gif" width="300" height="200"/>
</p>

<p align="center">
	<img src="/media/scene3.gif" width="300" height="200"/>
	<img src="/media/scene4.gif" width="300" height="200"/>
</p>

### Short Summary
**Goal**: Plan safe trajectories for a mobile robot using Rapidly-exploring Random Trees (RRT) combined with Control Barrier Functions (CBF). Basically, instead of computing the collision checking as in the standard RRT, we compute the control inputs by using the CBF to modify the provided primitives.

- Kinematic **model** of the unicyle mobile robot

<p align="center">
  <span>  $$ \begin{bmatrix} \dot{x} \\ \dot{y} \\ \dot{\theta} \end{bmatrix} = \begin{bmatrix} \cos{\theta} \\ \sin{\theta} \\ 0 \end{bmatrix}v + \begin{bmatrix} 0 \\ 0\\ 1 \end{bmatrix}\omega$$</span><br>
</p>

- Choice of the **CBF** 
<p align="center">
  <span>  $$ \mathcal{C}_s := \{x \in \mathbb{R}^2 : \sqrt{(x-x_{obs})^2 + (y - y_{obs})^2} \geq \tau \} $$</span><br>
</p>

- RRT vs RRT + CBF **algorithm** comparison <br/>
<p align="center">
	<img src="/media/RRT_algo.png" width="350" height="350"/>
	<img src="/media/RRT_CBF_algo.png" width="350" height="350"/>
</p>

<small>Last Update: June, 2021</small>

## Aircraft Landing Control <span id="aircraft-landing-control" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://github.com/Emanuele-n/Aircraft-Landing-Gear-Simulation-and-Control) </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://github.com/Emanuele-n/Aircraft-Landing-Gear-Simulation-and-Control/blob/main/Vehicle_Systems_Dynamics.pdf)</span>
<p align="center">
	<img src="/media/landing.png" width="550" />
</p>
**GOAL**: Design a control system for the landing gear of an aircraft to ensure a safe and autonomous landing. 
- The aircraft is **modeled** as a 7-DOF system with state variable $$x = [ z, z_{wf}, z_{wr}, \theta, x, \omega_{f}, \omega_{r} ]$$. The controlled elements are the elevators, the flaps and the active suspension of the front landing gear, together with the breaks of the rear and front gear, yelding to the control input $$ u = [ L, u_{\theta}, F_{aereo}, C_r, C_f ]$$. 
<p align="center">
	<img src="/media/aircraft_variables.png" width="300" />
</p>
<p align="center">
  <span>  $$  \ddot{z} = -\dfrac{1}{M}\left[ k_f(z_f-z_{wf})+c_f(\dot{z}_{f}-\dot{z}_{wf})+k_r(z_r-z_{wr} )+c_r (\dot{z}_{r}-\dot{z}_{wr}) + L \right ]-g $$
  $$ \ddot{z}_{wf} = \dfrac{1}{m_t}\left[ k_f(z_f-z_{wf})+c_f(\dot{z}_{f}-\dot{z}_{wf})-k_t(z_{wf}-y_f ) \right ]-g  $$
  $$ \ddot{z}_{wr} = \dfrac{1}{m_t}\left[ k_r(z_r-z_{wr})+c_r(\dot{z}_{r}-\dot{z}_{wr})-k_t(z_{wr}-y_r ) \right ]-g $$
  $$ \ddot{\theta} = \dfrac{1}{J}\left[  -(k_f l_f) (z_f-z_{wf}) -(c_f l_f) (\dot{z}_{f} - \dot{z}_{wf})+ (k_r l_r) (z_r-z_{wr}) + (c_r l_r) (\dot{z}_{r} - \dot{z}_{wr}) +u_{\theta} \right] $$
  $$ \ddot{x} = \dfrac{1}{M}(F_{long_{r}} + F_{long_{f}} - F_{aereo}) $$
  $$ \dot{\omega}_{r} = \dfrac{1}{I_{wr}}(C_r - F_{long_{r}R_{t}} - C_{roll_{r}}) $$
  $$ \dot{\omega}_{f} = \dfrac{1}{I_{wf}}(C_f - F_{long_{f}R_{t}} - C_{roll_{f}}) $$
  </span><br>
</p>
- The control system is designed to dived the problem in two pahses: the **approach** and the **touchdown**. <br/>
The first one, aims to regulate the height $$z_d $$ and the angle $$\theta_d$$ by means of the lift force $$ L $$ and it is controlled by a **PID** controller joint with a **Feedback Linearization**. <br/>
The second one aim to regulate the horizontal velocity $$ \dot{x} $$ and the angle $$\theta_d$$ by using the breaks $$  C_r, C_f $$ and the active suspension $$ u_{\theta}$$, this phase is controlled with a Suspension Variational Feedback Controller.

<small> Last Update: July, 2021 </small>

## Trajectory Tracking of a KUKA LBR 7R <span id="trajectory-tracking-of-a-kuka-lbr-7r" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://github.com/Emanuele-n/Robot-Learning-Control)</span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://github.com/Emanuele-n/Robot-Learning-Control/blob/main/Robotics_II.pdf) </span>
<p align="center">
	<img src="/media/kuka.png" height="150" />
	<img src="/media/gpr_q1.png" height="200" />
</p>
**GOAL**: When performing a Feedback Linearization (FL) there are always mismatches between the model and the real system, the goal of this porject is to use the Gaussian Process Regression (GPR) to learn the mismatch and to improve the tracking performance of the robot. <br/>
- Robot dynamic **model** 
<p align="center">
  <span>  $$ M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = u $$</span><br>
</p>
- The **control** law is obtained by using the FL plus a learned correction factor $$ \delta u $$ from the GPR
<p align="center">
  <span>  $$ u = M(q)(\ddot{q} + K_p (q_d - q) + K_d (\dot{q}_d - \dot{q}) ) + C(q, \dot{q})\dot{q} + G(q) + \delta u $$</span><br>
</p>
- Training the GPR with the **data** collected from the robot
<p align="center">
  <span>  $$ x_i = [q_i , \dot{q}_i , \ddot{q}_i ]$$
  $$ y_i = \delta u_i $$
  </span><br>
  With the training data we can compute the mean and the variance of the prediction. <br>
  Here we used the radial basis function kernel
  <span> $$ k(x,x') = \sigma^2 \exp(-\dfrac{1}{2}\norm{x-x'}^2_W) $$</span><br>
  and the Logarithmic Marginal Likelihood with sparsification as the loss function
  <span> $$     log\ p(\mathbf{y}|X,\boldsymbol{\theta}) =
    (n-m)log(\sigma_n)
    + \sum_{i=1}^m log(l_{M,ii})
    + \frac{1}{2\sigma^2}(\mathbf{y}^T\mathbf{y} - \boldsymbol{\beta}_I^T\boldsymbol{\beta}_I)
    + \frac{n}{2} log (2\pi)
    + \frac{1}{2\sigma^2} trace(K - V^T V) $$</span><br>


</p>

<small> Last Update: July, 2020 </small>




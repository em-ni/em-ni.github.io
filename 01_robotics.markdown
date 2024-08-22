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
- mfc
- 
 -->

Here are some of the projects I've been involved in the field of Control, Robotics and Machine Learning. 

## Robotics & Control Projects
- PhD Related
  - [Vision-Based Robotic Bronchoscope Localization and Cancer Detection](#vision-based-robotic-bronchoscope-localization-and-cancer-detection)
  - [Contact Force Model for Soft Robots](#contact-force-model-for-soft-robots)
- Work Related
  - [Panda for Hyperthermia](#panda-for-hyperthermia)
- Master's Thesis
  - [MPC for Soft Robots](#mpc-for-soft-robots)
- Master's Projects
  - [Motion Planning with RRT + CBF](#motion-planning-with-rrt--cbf)
  - [Aircraft Landing Control](#aircraft-landing-control)
  - [Trajectory Tracking of a KUKA LBR 7R](#trajectory-tracking-of-a-kuka-lbr-7r)
  - [Optimal Control of Covid-19 Pandemic](#optimal-control-of-covid-19-pandemic)
  - [Control of a discrete time Mass-Spring-Damper system with input delay](#control-of-a-discrete-time-mass-spring-damper-system-with-input-delay)
  - [Robot learning techniques and MPC for controlling robots with nonlinear flexibility](#robot-learning-techniques-and-mpc-for-set-point-regulation-of-robots-with-nonlinear-flexibility-on-the-joints)
- Master's Homeworks
  - [Cooperative attitude synchronization in satellite swarms: a consensus approach](#cooperative-attitude-synchronization-in-satellite-swarms-a-consensus-approach)
  - [A New Method for the Nonlinear Transformation of Means and Covariances in Filters and Estimators](#a-new-method-for-the-nonlinear-transformation-of-means-and-covariances-in-filters-and-estimators)
  - [DC-motor and 4-tanks system control with 
  <span>  $$  H_\infty $$ and $$ \mu $$-synthesis techniques</span>
](#dc-motor-and-4-tank-system-control)
  - [Implementation of CNN, LSTM and free design of a neural network architecture](#implementation-of-cnn-lstm-and-free-design-of-a-neural-network-architecture)

<br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br>

## Vision-Based Robotic Bronchoscope Localization and Cancer Detection <span id="vision-based-robotic-bronchoscope-localization-and-cancer-detection" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>]() </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>]()</span>

<p align="center">
  <img src="/media/fpv_nav.gif" width="360" />
  <img src="/media/tpv_nav.gif" width="360" /><br>
  Simulated example of input camera images and the corresponding output of the realtime robot localization 
</p>
<br>
work in progress...

<br><br><br>

## Contact Force Model for Soft Robots <span id="contact-force-model-for-soft-robots" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://github.com/Emanuele-n/sim) </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>]()</span>
Soft Robots find the best application to robotic surgery, where the compliance of the structure is a key feature to ensure the safety of the patient. However, the compliance of the robot makes the control more challenging, especially when the robot is in contact with the environment and it doesn't have force sensors. <br/>
Here I developed a contact force model for a soft continuum robot, which is able to predict the contact force and the deformation of the robot when in contact with an obstacle. <br/>
<p align="center">
	<img src="/media/robot_force.png" height="150" height="200"/>
	<img src="/media/sofa_force_sim.gif" height="100" height="200"/>
	<!-- <img src="/media/force_model_vs_data.png" height="150" height="200"/> -->
</p>

### Short Summary
**Goal**: Develop a contact force model for a soft robot in contact with the environment
- First, start with modelling the actuators to get the robot dependent mapping, which gives the robot **kinematics** from the pressure input, together with solving a shape optimization problem to find the best geometric parameters (e.g. shape of the deformed tube)
<p align="center">
  <span>  $$  T( q (p) ) =   \small
    \begin{bmatrix}            
        -\sin{q (p)} & \cos{q (p)} & 0 & L^o  \dfrac{(1 - \cos{q (p)})}{q (p)}\\[2.5mm]
        \cos{q (p)} & \sin{q (p)} & 0 & L^o \dfrac{\sin{q (p)}}{q (p)}\\[2.5mm]
        0 & 0 & 1 & 0\\[2.5mm]
        0 & 0 & 0 & 1\\[2.5mm]
    \end{bmatrix}
    \normalsize  $$</span><br>
</p>
- Then, use the **PCC** model to get the robot independent mapping which gives the robot **dynamics** and the **static** equilibrium 
<p align="center">
  <span>  $$  G(q) + Kq = u_{\tau}(p) +  J^T (q) \gamma_e $$</span><br>
</p>
- Finally, combine the static equilibrium with the axial force and tune the final coefficients to get the **contact force model**
<p align="center">
  <span>  $$  F_{c} = F_{\kappa} + \dfrac{\alpha F_{\epsilon} p}{q} + \beta $$</span><br>
</p>

<small> Last Update: August, 2024 </small>

<br><br><br>

## Panda for Hyperthermia <span id="panda-for-hyperthermia" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://github.com/Emanuele-n/panda-win)</span>
While working at [Medlogix](https://www.albahyperthermia.com/) I had the possibility of writing the code for controlling the [Panda](https://robodk.com/robot/Franka/Emika-Panda) robot from Franka Emika. Here I experimented with many different control techniques such as FBL, impedance and admittance control, to find the best solution for curing cancer with hyperthermia! <br/>
One day I asked chatGPT to write the parametric Cartesian equations of an heart and I made the robot follow the trajectory, basically I am just a bridge between two robots. 
<p align="center">
	<img src="/media/love_trajectory.gif"/><br/> 
Here is the Panda spreading some love
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



<br><br><br>

## MPC for Soft Robots <span  id="mpc-for-soft-robots" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://www.mathworks.com/matlabcentral/fileexchange/104060-soropcc)</span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://drive.google.com/file/d/1wzDfhW_K8pfrNxatST2SaEYtEBRyG7u8/view?usp=share_link)</span>


While I was writing my Master's [thesis](https://drive.google.com/file/d/1wzDfhW_K8pfrNxatST2SaEYtEBRyG7u8/view?usp=share_link) I decided to share the code I was using as a new MATLAB [Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/104060-soropcc). SoRoPCC  allows you to simulate and control a customizable soft manipulator, both for Shape Control and for Task-space Control, either by controlling the curvature of each CC segment or the position and orientation of the tip of the robot. Moreover it is also embedded with the MPC Toolbox, so you can control the robot using the Model Predictive Control.<br/>
Here are some examples of trajectory tracking and set-point regulation of a 3-DOF tentacle.

<p align="center">
	<img src="/media/track_fully.gif" width="350" height="300"/>
	<img src="/media/track_under.gif" width="350" height="300"/>
</p>

<p align="center">
	<img src="/media/track_plus_alpha.gif" width="350" height="300"/>
	<img src="/media/reg_obstacle.gif" width="350" height="300"/><br>
	Top-left is fully actuated; top-right is underactuated on the last CC segment; bottom-left shows how it is possible to control also the orientation; bottom-right is an attempt of sliding on an obstacle to exploit the compliance structure of the robot.
</p>



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


<br><br><br>

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

<br><br><br>

## Aircraft Landing Control <span id="aircraft-landing-control" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://github.com/Emanuele-n/Aircraft-Landing-Gear-Simulation-and-Control) </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://github.com/Emanuele-n/Aircraft-Landing-Gear-Simulation-and-Control/blob/main/Vehicle_Systems_Dynamics.pdf)</span>
<p align="center">
	<img src="/media/landing.png" width="550" />
</p>

### Short Summary
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

<br><br><br>

## Trajectory Tracking of a KUKA LBR 7R <span id="trajectory-tracking-of-a-kuka-lbr-7r" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://github.com/Emanuele-n/Robot-Learning-Control)</span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://github.com/Emanuele-n/Robot-Learning-Control/blob/main/Robotics_II.pdf) </span>
<p align="center">
	<img src="/media/kuka.png" height="150" />
	<img src="/media/gpr_q1.png" height="200" />
</p>

### Short Summary
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



<br><br><br>

## Optimal Control of Covid-19 Pandemic <span id="optimal-control-of-covid-19-pandemic" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://drive.google.com/file/d/1uI6YFrqJ_GbqG4mblhvyfIa9R_5uUN2x/view?usp=sharing)</span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://drive.google.com/file/d/1uoJsdC5bN9xC22uGGpPnZg7gT6mW5ZJL/view?usp=sharing)</span>

<p align="center">
  <img src="/media/SEIR_plot.png" height="250" />
  <img src="/media/covid_results.png" height="250" />
</p>

### Short Summary
**GOAL**: Design an optimal control strategy to minimize the number of infected and loss people due to the lockdown. <br/>
- The **model** is a SEIR model $$ x = [S,E,I,R] $$ with the addition of the lockdown control $$ u(t) $$ and health care capacity $$ v(t) $$

<p align="center">
  <span>  $$ \begin{align*}
  \frac{dS}{dt} &= n - (1 - u) \beta \frac{S I}{N} - mS \\
  \frac{dE}{dt} &= (1 - u) \beta \frac{S I}{N} - (\alpha + m)E \\
  \frac{dI}{dt} &= \alpha E - (\epsilon + \gamma + m + v)I \\
  \frac{dR}{dt} &= (\gamma + v)I - mR
  \end{align*}
  $$
  </span><br>
</p>

- The **cost function** used is though to reduce as much as possible the number of infected and exposed, while also avoiding too strict lockdowns and overloading the health care system
<p align="center">
  <span>  $$ J = \int_{t_0}^{t_f} \left\{ E(t) + I(t) + \frac{A_1}{2} u^2(t) + \frac{A_2}{2} v^2(t) \right\} dt
 $$</span><br>
</p>

- By applying the Pontryagin principle we can find the optimal control law $$ u^* $$ and $$ v^* $$, which are the solution of the Hamiltonian system, and the resulting closed loop system is
<p align="center">
  <span>  $$ \begin{align*}
  S_{i+1} &= \frac{S_i + h n \frac{I_i}{1 + h(m + (1 - u^*_i)\beta \frac{I_i}{N})}}{1 + h m} \\
  E_{i+1} &= \frac{E_i + h (1 - u^*_i) \beta \frac{S_{i+1} I_i}{N}}{1 + h(\alpha + m)} \\
  I_{i+1} &= \frac{I_i + h (\alpha E_{i+1})}{1 + h(\epsilon + \gamma + m + v^*_i)} \\
  R_{i+1} &= \frac{R_i + h ((\gamma + v^*_i) I_{i+1})}{1 + hm} \\
  \lambda_{n-i-1}^{1} &= \frac{\lambda_{n-i}^{1} + h ((1 - u^*_i) \lambda_{n-i}^{2} \beta \frac{I_{i+1}}{N})}{1 + h d + h(1 - u^*_i) \beta \frac{I_{i+1}}{N}} \\
  \lambda_{n-i-1}^{2} &= \frac{\lambda_{n-i}^{2} + h(1 + \alpha \lambda_{n-i}^{3})}{1 + h(\alpha + m)} \\
  \lambda_{n-i-1}^{3} &= \frac{\lambda_{n-i}^{3} + h(1 + \lambda_{n-i}^{1} (1 - u^*_i) \beta \frac{S_{i+1}}{N} + (\gamma + v^*_i) \lambda_{n-i}^{4})}{1 + h(\epsilon + \gamma + d + v^*_i)} \\
  \lambda_{n-i-1}^{4} &= \frac{\lambda_{n-i}^{4} + h m}{1 + h m} \\
  M_{i+1} &= (\lambda_{n-i-1}^{1} - \lambda_{n-i}^{1}) \frac{\beta I_{i+1} S_{i+1}}{N A_1} \\
  r_{i+1} &= (\lambda_{n-i-1}^{3} - \lambda_{n-i}^{4}) \frac{I_{i+1}}{A_2} \\
  u^*_{i+1} &= \min(1, \max(0, M_{i+1})) \\
  v^*_{i+1} &= \min(1, \max(0, r_{i+1}))
  \end{align*}
  $$
  </span><br>
</p>

<small> Last Update: July, 2020 </small>


<br><br><br>

## Control of a discrete time Mass-Spring-Damper system with input delay <span id="control-of-a-discrete-time-mass-spring-damper-system-with-input-delay" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://drive.google.com/file/d/1_zs_uU1a59uJ8aZk0RwDxtk4zw2WYfK3/view?usp=sharing) </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://drive.google.com/file/d/1_zs_uU1a59uJ8aZk0RwDxtk4zw2WYfK3/view?usp=sharing)</span>

The mass-spring-damper system is a classic example in control theory, and it is often used to demonstrate the behavior of different control strategies. In this project, I explored the control of a discrete-time mass-spring-damper system with input delay

<p align="center">
  <img src="/media/mass_spring_damper.gif" height="200" />
  <img src="/media/dcs_results.png" height="235" />
</p>

### Short Summary
**Goal**: The goal was to design a controller that could stabilize the system and ensure that it behaves as desired, despite the delay in the input signal.

- **Model**: The system is modeled as a discrete-time mass-spring-damper system with the input delayed by $$d$$ time steps.
<p align="center">
  <span>  $$ \begin{equation}
\begin{bmatrix}
\dot{x}_1 \\
\dot{x}_2
\end{bmatrix} =
\begin{bmatrix}
0 & 1 \\
-\frac{k}{m} & -\frac{b}{m}
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix} +
\begin{bmatrix}
0 \\
\frac{1}{m}
\end{bmatrix}
F(t-d)
\end{equation}
 $$</span><br>
</p>

- The **control** strategy involved designing a state feedback controller to stabilize the system and ensure that it behaves as desired, regardless of the input delay. here I have explored Predictor-Preview Controllers, in particular focusing on Switchd-Low-Gain Feedback

<small> Last Update: December, 2020 </small>

<br><br><br>

## Robot learning techniques and MPC for set point regulation of robots with nonlinear flexibility on the joints <span id="robot-learning-techniques-and-mpc-for-set-point-regulation-of-robots-with-nonlinear-flexibility-on-the-joints" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://github.com/caciolai/Robot-learning-techniques-for-set-point-regulation-of-robots-with-nonlinear-flexibility-on-the-joint) </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://github.com/caciolai/Robot-learning-techniques-for-set-point-regulation-of-robots-with-nonlinear-flexibility-on-the-joint/blob/main/documentation/report.pdf)</span>

Set-point regulation for flexible joints robot is a classic control problem in underactuated robotics. In fact, the noncollocation of the torque inputs and the elastic coupling between the motor and the link makes this kind of system very challenging to control, with respect to the rigid ones.

<p align="center">
  <img src="/media/flexible_link.png" height="200" /><br/>
  The Spong model of a flexible link<br/> <br/> <br/> 
  <img src="/media/mpc.png" height="200" /> <br/>
  MPC strategy <br/> <br/> <br/> 

  <img src="/media/gpr.png" height="200" /> 
  <img src="/media/neural_network.png" height="200" /> <br/>
  Learning techniques tested, conceptual difference between GPR and NN
</p>

### Short Summary
**Goal**: The purpose of this project is to study how a data-driven control technique can be applied when the system model is not precisely known, exploiting a robot learning method developed at DIAG, analyzing its applicability and performance in this context. In particular, the latter is used to estimate the nonlinear elastic term of the robot model and combined with a Model Predictive Control (MPC) strategy to regulate the robot's joints to a desired set-point.

- The system is **modeled** as a flexible joint robot with a nonlinear elasticity term, so it can be divided into two subsystems coupled by the nonlinear elastic term.
<p align="center">
  <span>  $$ \begin{align*}
M(q) \ddot{q} + c(q, \dot{q}) + g(q) + \psi(q - \theta) + D \dot{q} &= 0 \\
B \ddot{\theta} - \psi(q - \theta) + D \dot{\theta} &= \tau
\end{align*}

$$</span><br>
</p>

- The **control** strategy involves using a Model Predictive Control (MPC) approach with tail cost formulation. In particular, the tail is chosen as the terminal cost of the constrained linear-quadratic regulator (LQR) computed on the linearized system.
<p align="center">
  <span>  $$ V(x, u) = \sum_{i=0}^{N-1} \ell(x(i), u(i)) + \sum_{k=N-1-K}^{N-1} V_f(x(k), u(k))
 $$</span><br> Getting the control input by minimizing the cost function
 <span> $$ \begin{align*}
\text{minimize } & V(x, u) \\
\text{subject to } & x(k+1) = f(x(k), u(k)) \\
& h(x, u) \leq 0
\end{align*}
$$</span><br>
</p>

- The **learning technique** used to estimate the nonlinear elasticity term of the system is the Gaussian Process Regression (GPR), which turned out to give better performance than standard machine learning techniques, due to its ability to imporove without overfitting while increasing the number of data points sampled from a fixed size workspace, such as for the case of a manipulator. <br>Once the kernel function $$ k(\cdot, \cdot) $$ is chosen, the mean and the variance of the prediction are given by
<p align="center">
  <span>  $$ f(x^*) = k^{T}(K)^{-1}y
 $$ 
 $$ V(x^*) = k(x^*, x^*) - k^{T}(K)^{-1}k
 $$</span><br>
</p>

<small> Last Update: June, 2021 </small>




<br><br><br><br><br><br>
## Other Projects
Here are minor projects, mostly theorethical, that I have been working on during my studies.
<br><br><br>

## Cooperative attitude synchronization in satellite swarms: a consensus approach <span id="cooperative-attitude-synchronization-in-satellite-swarms-a-consensus-approach" style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://docs.google.com/presentation/d/1RRxgozICgrMBlWLFej99_7suD8wGwpvX/edit?usp=sharing&ouid=117079509386533656063&rtpof=true&sd=true) </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://doi.org/10.3182/20070625-5-FR-2916.00039)</span>

<p align="center">
  <img src="/media/satellite_swarm.gif" height="200" />
</p>

**Problem**: Autonomous synchronization of attitudes in a swarm of spacecraft

**Solution**: Two types of control laws, in terms of applied control torques, that globally drive the swarm towards attitude synchronization
- Solution 1: requires tree-like or all-to-all inter-satellite communication (most efficient)
- Solution 2: works with nearly arbitrary communication (more robust)

<small> Last Update: June, 2021 </small>

<br><br><br>

## A New Method for the Nonlinear Transformation of Means and Covariances in Filters and Estimators <span id="a-new-method-for-the-nonlinear-transformation-of-means-and-covariances-in-filters-and-estimators" style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://drive.google.com/file/d/1t4AP8CL7wIqtvG9lRDwv4nckg7gKg5Ls/view?usp=sharing) </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](10.1109/9.847726)</span>

The goal of this project was to make a report, finding the novelty introduced in the paper. 
This paper presents a novel method for generalizing the Kalman filter to handle nonlinear systems effectively. Unlike the traditional Extended Kalman Filter (EKF) that relies on linearization, this approach uses a set of deterministically selected samples (sigma points) to parameterize the mean and covariance of a distribution, thus allowing a more accurate and straightforward implementation without the need for Jacobians. The method provides a significant improvement over EKF, particularly in systems with pronounced non-linear characteristics where EKF may introduce substantial biases or errors. The new approach is demonstrated through simulations showing enhanced prediction accuracy and robustness in tracking and estimation tasks.

### Short Summary
**Goal**: Develop a generalized Kalman filter method for nonlinear systems without relying on linearization.

**Results**: Uses sigma points to capture the true mean and covariance of the state distribution, avoiding the need for linear approximations.
Demonstrated superior performance to EKF, especially in handling non-linearities.
Provides a more straightforward implementation approach, as it does not require the computation of Jacobians.

<small>Last Update: June, 2020</small>

<br><br><br>

## DC-motor and 4-tanks system control with <span>  $$  H_\infty $$ and $$ \mu $$-synthesis techniques</span> <span id="dc-motor-and-4-tank-system-control" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://drive.google.com/drive/folders/1SvuPmHNxdLnSUjKJ_k6mfa5pIsZ3yIAx?usp=sharing) </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://drive.google.com/drive/folders/1SvuPmHNxdLnSUjKJ_k6mfa5pIsZ3yIAx?usp=sharing)</span>

This is a veeery long series of homework aimed at specializing in the control of MIMO systems with the $$ H_\infty $$ and $$ \mu $$-synthesis techniques. The first part of the project is about the control of a DC-motor, while the second part is about the control of a 4-tanks system. 

<small>Last Update: June, 2021</small>

<br><br><br>

## Implementation of CNN, LSTM and free design of a neural network architecture <span id="implementation-of-cnn-lstm-and-free-design-of-a-neural-network-architecture" style="margin-left: 10px;">[<i class="fa fa-file-code-o"></i>](https://drive.google.com/drive/folders/1CjRv3hZ0HUbIk4pZWGAoHemNq2NZXw-g?usp=sharing) </span> <span style="margin-left: 10px;">[<i class="fa fa-file-pdf-o"></i>](https://drive.google.com/drive/folders/1CjRv3hZ0HUbIk4pZWGAoHemNq2NZXw-g?usp=sharing)</span>

A series of homeworks aimed at specializing in the implementation of neural networks, in particular Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks. The last part of the project is about the free design of a neural network architecture.

<small>Last Update: January, 2020</small>
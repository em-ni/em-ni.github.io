---
layout: page
title: Software
permalink: /software/
---

<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PQFC01D0LX"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PQFC01D0LX');
</script>

## PCC Soft Robots
While I was writing my Master's [thesis](https://drive.google.com/file/d/1wzDfhW_K8pfrNxatST2SaEYtEBRyG7u8/view?usp=share_link) I decided to share the code I was using as a new MATLAB [Toolbox](https://www.mathworks.com/matlabcentral/fileexchange/104060-soropcc). SoRoPCC  allows you to simulate and control a customizable soft manipulator, both for Shape Control and for Task-space Control, either by controlling the curvature of each CC segment or the position and orientation of the tip of the robot. Moreover it is also embedded with the MPC Toolbox, so you can control the robot even with the Model Predictive Control.<br/>
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



## Motion Planning with RRT + CBF 
[Here](https://github.com/Emanuele-n/Enhancing-kinodynamic-RRT-using-CBF-based-steering) is a project about Probabilistic Motion Planning of Unicyle Mobile Robot using Rapidly-exploring Random Tree (RRT) with Control Barrier Functions (CBF)

Here are some examples of the application of the algorithm in four scenes with increasing difficulty
<p align="center">
	<img src="/media/scene1.gif" width="300" height="200"/>
	<img src="/media/scene2.gif" width="300" height="200"/>
</p>

<p align="center">
	<img src="/media/scene3.gif" width="300" height="200"/>
	<img src="/media/scene4.gif" width="300" height="200"/>
</p>
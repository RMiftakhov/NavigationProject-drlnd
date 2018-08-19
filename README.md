# The solution for the Unity Banana environment (project 1 - Navigation)
![Banana-gif](https://github.com/RMiftakhov/NavigationProject-drlnd/blob/master/banana-gif.gif)
The code is based on materials from Udacity Deep Reinforcement Learning Nanodegree Program.

## Project Details
The Unity Banana simulates an environment where an agent has to learn to collect yellow and avoid blue bananas. The corresponding reward is +1 for yellow and -1 for blue bananas.

To navigate the world, the agent has four discrete actions to chose from:
* 0 - move forward
* 1 - move backward
* 2 - turn left
* 3 - turn right.

The environment returns state vector that consists of 37 values, which comprises an agent's velocity, and ray-based perception object in agent's forward direction.     

The environment considered as being solved if an agent would get an average of +13 reward over 100 consecutive episodes.

## Getting Started
Download the Unity environment from one of the links below.
* [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

In order to train the model or inference the computed weights, the following python packages need to be installed:
* *pytorch*
* *unityagents*
* *numpy*
* *matplotlib* 

## Instructions
Since the repository provides the jupyter notebook, follow the steps of execution.

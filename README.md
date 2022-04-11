#  <img src="./readme_stuff/icon_1024.png" height= 70 align="left" />Beagle

Beagle is a team project made by Anderson Addo and Alana White of Brown University to explore shader evolution and L-system trees. It was built for [CS123](https://cs.brown.edu/courses/cs123) ([stencil](https://github.com/cs123tas/final-stencil)). It's named after Darwin's HMS Beagle.

It was built with Qt Creator (v 4.11). 

**You can get a MacOX build for it in the "Releases" section of this Github Repo!**

By the way, here was our team banner!

<img src="./readme_stuff/team_banner.jpg" height=300 />

Oh, any by the way, we tried to compile this to WebAssembly, but it proved [too troublesome](https://forum.qt.io/topic/121724/qt-webassembly-unable-to-build-redefinition-of-__glewcreateprogram-as-different-kind-of-symbol).

## The 4 Parts

### The Shader Tab

The shader tab is where you can use the laws of evolution to make cool looking shaders. You get to play God with the shaders.

To get started with the shader tab, be sure to press the "Initialize!" button: this only needs to be pressed once. Also press the "Show src" button whenever you want to be able to analyze the source code of shaders that you find interesting. Keep in mind that the source you see there is only the critical part of the source. They do reference to custom methods and whatnot; a quick look at the source code of the project and you'll see the definition of those functions. It also only shows the fragment shader; you'll probably want to see the vertex shader too (it's a super simple vertex shader though).

Regardless, once you've  initialized the scene, there's a number of things you can do. 

1. If you don't like some/all of the shaders that you see, you can replace them with completely new genotypes by selecting those shaders in the selection pane and pressing the "Refresh Population" button (in the "Replace Shaders" tab.)
2. You can also decide to replace the shaders that you don't like with mutations of another shader that you do like. Select the shaders you want to replace in the selection pane, then select the id of the "donor" shader, and hit the "replace with mutations" button.
3. You can also mutate all the currently selected shaders in place by going to the "Mutate current shaders" tab and pressing the "Mutate" button.
4. You can also decide to replace shaders with the offspring of other shaders! This is my favorite feature, actually. Go to the "create offspring" tab, select the ids of the parents, select the shaders you want to replace in the selection pane, and hit the "crease offspring" button!

##### A little note on generations

So shaders normally start out at generation [0] and increment their generation with every mutation. If any part of their source code is edited when they go to generation [n], then that part of the code will have the [n] tab by it.

Offspring, though, start out at generation 1. This is because they get all their initial genes from one parent at generation 0, then go though a custom mutation (which increases their generation) which mutates it with genes from the second parent.



### The Import Tab

This tab was made to allow people to paste in shaders that they liked in the past and show them off or tweak them. If you'd like a few to get started, have a look at these!



<details><summary>Fav Shader 1</summary>
<p>

```
average(my_cross(transplantZ(min(vec3(-0.48548174+0,0.63039374+0,-0.67468715+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0) - vec3(pos.y+0, pos.y+0, pos.y+0) + min(vec3(0.29284927+0,0.87573367+0,1.181393+0),  vec3(timevar+0, timevar+0, timevar+0))),  perlinNoiseVec3((transplantX((vec3(pos.y+0, pos.y+0, pos.y+0) / vec3(timevar+0, timevar+0, timevar+0)),  fractal(average(vec3(timevar+0, timevar+0, timevar+0),  vec3(0.57004356+0,1.2222601+0,1.4275744+0)),  vec3(1.4275744+0,1.6760367+0,-0.30855873+0), vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(timevar+0, timevar+0, timevar+0), true, true, vec3(pos.y+0, pos.y+0, pos.y+0), 4, 4, 3, 1, 1, 1, 2, 4, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1))* transplantX(vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(pos.y+0, pos.y+0, pos.y+0))),  vec3(pos.x+ 0, pos.x+0, pos.x+0))),  fractal(atan(vec3(timevar+0, timevar+0, timevar+0)),  vec3(timevar+0, timevar+0, timevar+0), average(max(vec3(timevar+0, timevar+0, timevar+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0)),  vec3(pos.z+0, pos.z+0, pos.z+0)),  transplantZ(vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(pos.z+0, pos.z+0, pos.z+0)), false, false, vec3(1.5938786+0,-0.39091733+0,1.5319189+0), 3, 3, 2, 1, 1, 0, 0, 1, 0, 1, 1, 0, 3, 3, 4, 1, 0, 0)),  (vec3(1.0299773+0,0.75646037+0,0.85278189+0)* fractal((vec3(pos.x+ 0, pos.x+0, pos.x+0) / max(vec3(timevar+0, timevar+0, timevar+0),  (perlinNoiseVec3(abs(vec3(timevar+0, timevar+0, timevar+0)),  transplantY(vec3(pos.y+0, pos.y+0, pos.y+0),  vec3(timevar+0, timevar+0, timevar+0)))* min(vec3(0.076585248+0,0.83288985+0,-0.26071259+0),  transplantZ(average(vec3(pos.y+0, pos.y+0, pos.y+0),  transplantY(vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(pos.z+0, pos.z+0, pos.z+0))),  vec3(1.1732373+0,0.5403502+0,0.9053238+0)))))),  (transplantX(transplantX(vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(pos.z+0, pos.z+0, pos.z+0)),  vec3(pos.y+0, pos.y+0, pos.y+0))* transplantY(vec3(1.5973549+0,0.33649653+0,1.1407388+0),  atan(transplantY(vec3(timevar+0, timevar+0, timevar+0),  vec3(timevar+0, timevar+0, timevar+0))))), sin(vec3(pos.x+ 0, pos.x+0, pos.x+0)) - fractal(fractal((vec3(pos.y+0, pos.y+0, pos.y+0)* vec3(pos.z+0, pos.z+0, pos.z+0)),  vec3(pos.x+ 0, pos.x+0, pos.x+0), vec3(pos.z+0, pos.z+0, pos.z+0),  abs(vec3(pos.x+ 0, pos.x+0, pos.x+0)), true, true, (vec3(pos.y+0, pos.y+0, pos.y+0) / vec3(pos.x+ 0, pos.x+0, pos.x+0)), 2, 1, 0, 1, 1, 1, 1, 0, 4, 0, 0, 1, 0, 0, 2, 1, 1, 0),  vec3(timevar+0, timevar+0, timevar+0), vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0), false, false, my_cross(perlinNoiseVec3(vec3(-0.65977585+0,-0.0086404383+0,1.5156628+0) + vec3(pos.x+ 0, pos.x+0, pos.x+0),  min(vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0))),  vec3(pos.y+0, pos.y+0, pos.y+0) + vec3(pos.x+ 0, pos.x+0, pos.x+0)), 3, 2, 1, 0, 1, 1, 1, 0, 4, 0, 0, 1, 0, 0, 2, 1, 1, 0),  sin(vec3(pos.y+0, pos.y+0, pos.y+0)), true, true, vec3(pos.z+0, pos.z+0, pos.z+0), 0, 2, 3, 1, 1, 1, 1, 2, 3, 1, 1, 1, 2, 3, 1, 1, 0, 0)))
```

</p>
</details>

<details><summary>Fav Shader 2</summary>
<p>

```
cos((transplantZ(vec3(pos.y+0, pos.y+0, pos.y+0),  vec3(0.47181803+0,0.17196974+0,0.51834494+0)) / fractal(min(vec3(timevar+0, timevar+0, timevar+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0)),  abs(my_cross(average(vec3(timevar+0, timevar+0, timevar+0),  vec3(pos.z+0, pos.z+0, pos.z+0)),  vec3(-0.66001582+0,0.84839863+0,1.4276572+0))), sin(vec3(pos.x+ 0, pos.x+0, pos.x+0)),  abs(vec3(pos.z+0, pos.z+0, pos.z+0)), true, true, sin(transplantZ(max(vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(pos.y+0, pos.y+0, pos.y+0)),  fractal(vec3(pos.x+ 0, pos.x+0, pos.x+0),  (vec3(pos.x+ 0, pos.x+0, pos.x+0)* vec3(pos.x+ 0, pos.x+0, pos.x+0)), vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(pos.y+0, pos.y+0, pos.y+0), true, true, vec3(pos.y+0, pos.y+0, pos.y+0), 0, 4, 3, 0, 0, 0, 2, 2, 2, 1, 0, 1, 3, 0, 0, 1, 1, 1))), 0, 4, 1, 1, 1, 1, 0, 2, 3, 1, 1, 1, 3, 1, 4, 0, 1, 0)))
```

</p>
</details>

<details><summary>Fav Shader 3</summary>
<p>

```
fractal(sin(fractal(abs(vec3(pos.x+ 0, pos.x+0, pos.x+0)),  vec3(0.36220726+0,1.3940891+0,1.3773476+0), vec3(pos.y+0, pos.y+0, pos.y+0),  (vec3(pos.x+ 0, pos.x+0, pos.x+0) / fractal(average(transplantX(perlinNoiseVec3(vec3(pos.y+0, pos.y+0, pos.y+0),  average(vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(pos.y+0, pos.y+0, pos.y+0))),  vec3(pos.z+0, pos.z+0, pos.z+0)),  vec3(0.028243922+0,-0.6344083+0,0.10517785+0)),  vec3(-0.4617658+0,-0.16093819+0,-0.039006475+0), vec3(pos.z+0, pos.z+0, pos.z+0),  transplantZ(vec3(pos.y+0, pos.y+0, pos.y+0) - vec3(-0.32460633+0,0.56094122+0,1.582038+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0) - fractal(vec3(-0.61226934+0,-0.46454433+0,0.032795604+0),  vec3(-0.46454433+0,0.032795604+0,0.35751772+0), vec3(pos.y+0, pos.y+0, pos.y+0),  vec3(pos.z+0, pos.z+0, pos.z+0), true, true, vec3(pos.z+0, pos.z+0, pos.z+0), 2, 4, 0, 0, 0, 1, 1, 4, 3, 0, 1, 0, 4, 4, 1, 1, 1, 0)), true, true, vec3(pos.x+ 0, pos.x+0, pos.x+0), 0, 3, 1, 0, 0, 0, 1, 4, 0, 0, 1, 1, 4, 1, 1, 1, 0, 1)), true, true, atan(vec3(pos.z+0, pos.z+0, pos.z+0)), 4, 3, 2, 0, 0, 0, 0, 2, 4, 0, 0, 0, 3, 1, 4, 1, 1, 0)),  transplantX(perlinNoiseVec3(fractal(transplantX(perlinNoiseVec3(vec3(pos.y+0, pos.y+0, pos.y+0),  average(vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(pos.y+0, pos.y+0, pos.y+0))),  vec3(pos.z+0, pos.z+0, pos.z+0)),  transplantZ(vec3(pos.y+0, pos.y+0, pos.y+0) - vec3(-0.32460633+0,0.56094122+0,1.582038+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0) - fractal(vec3(-0.61226934+0,-0.46454433+0,0.032795604+0),  vec3(-0.46454433+0,0.032795604+0,0.35751772+0), vec3(pos.y+0, pos.y+0, pos.y+0),  vec3(pos.z+0, pos.z+0, pos.z+0), true, true, vec3(pos.z+0, pos.z+0, pos.z+0), 2, 4, 0, 0, 0, 1, 1, 4, 3, 0, 1, 0, 4, 4, 1, 1, 1, 0)), perlinNoiseVec3(vec3(pos.x+ 0, pos.x+0, pos.x+0),  (vec3(pos.x+ 0, pos.x+0, pos.x+0) - vec3(pos.z+0, pos.z+0, pos.z+0) / cos(transplantZ(vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0))))),  vec3(pos.x+ 0, pos.x+0, pos.x+0), false, false, vec3(0.028243922+0,-0.6344083+0,0.10517785+0), 3, 2, 2, 1, 0, 0, 2, 1, 4, 0, 1, 0, 1, 2, 4, 1, 1, 1),  vec3(pos.x+ 0, pos.x+0, pos.x+0) - vec3(pos.z+0, pos.z+0, pos.z+0)),  my_cross(transplantX(vec3(pos.z+0, pos.z+0, pos.z+0) + average(vec3(-0.51711076+0,0.83015585+0,0.897416+0),  vec3(timevar+0, timevar+0, timevar+0)),  abs(vec3(pos.x+ 0, pos.x+0, pos.x+0))),  atan(atan(max(vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(pos.y+0, pos.y+0, pos.y+0)))))), transplantZ(transplantZ(min(vec3(pos.z+0, pos.z+0, pos.z+0) - vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0)),  vec3(timevar+0, timevar+0, timevar+0)),  vec3(pos.x+ 0, pos.x+0, pos.x+0)),  min(transplantZ(transplantY(vec3(timevar+0, timevar+0, timevar+0),  vec3(pos.z+0, pos.z+0, pos.z+0)),  vec3(pos.z+0, pos.z+0, pos.z+0)),  abs(transplantZ(vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(pos.y+0, pos.y+0, pos.y+0)))), false, false, vec3(pos.y+0, pos.y+0, pos.y+0), 0, 4, 3, 1, 0, 0, 4, 0, 2, 1, 0, 0, 4, 4, 3, 1, 1, 1)
```

</p>
</details>


<details><summary>Fav Shader 4</summary>
<p>

```
fractal(sin(fractal(vec3(timevar+0, timevar+0, timevar+0),  vec3(-0.12119874+0,0.1405447+0,-0.76657218+0), vec3(pos.z+0, pos.z+0, pos.z+0),  atan(vec3(0.011142236+0,0.0016521072+0,-0.096066147+0)), true, true, transplantX(vec3(pos.y+0, pos.y+0, pos.y+0),  vec3(timevar+0, timevar+0, timevar+0)), 4, 1, 3, 1, 0, 0, 4, 3, 2, 0, 1, 1, 2, 3, 4, 1, 0, 0)),  transplantX(perlinNoiseVec3(vec3(0.32939366+0,0.25905618+0,0.25072694+0),  vec3(pos.z+0, pos.z+0, pos.z+0)),  vec3(pos.z+0, pos.z+0, pos.z+0)), sin(atan(vec3(pos.x+ 0, pos.x+0, pos.x+0))),  min(vec3(pos.z+0, pos.z+0, pos.z+0),  abs(transplantZ(vec3(pos.z+0, pos.z+0, pos.z+0),  min(vec3(0.32939366+0,0.25905618+0,0.25072694+0),  vec3(timevar+0, timevar+0, timevar+0))))), false, false, transplantZ(min(vec3(pos.z+0, pos.z+0, pos.z+0) - vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0)),  vec3(timevar+0, timevar+0, timevar+0)), 0, 4, 3, 1, 0, 0, 4, 0, 2, 1, 0, 0, 4, 4, 3, 1, 1, 1)
```

</p>
</details>

<details><summary>Fav Shader 5</summary>
<p>

```
(my_cross((fractal(vec3(timevar+0, timevar+0, timevar+0),  average(vec3(pos.y+0, pos.y+0, pos.y+0),  vec3(pos.z+0, pos.z+0, pos.z+0) + vec3(timevar+0, timevar+0, timevar+0)), vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0), true, true, transplantX(vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(timevar+0, timevar+0, timevar+0)), 2, 0, 1, 0, 1, 0, 0, 2, 4, 1, 1, 1, 1, 3, 1, 0, 0, 1)* vec3(pos.x+ 0, pos.x+0, pos.x+0) - vec3(0.043860868+0,1.5918975+0,0.37197471+0)),  vec3(pos.z+0, pos.z+0, pos.z+0)) / vec3(pos.x+ 0, pos.x+0, pos.x+0))
```

</p>
</details>

<details><summary>Fav Shader 6</summary>
<p>

```
fractal((average(vec3(0.24134713+0,0.35078835+0,0.35297719+0),  transplantX(vec3(pos.x+ 0, pos.x+0, pos.x+0),  atan(vec3(pos.x+ 0, pos.x+0, pos.x+0))))* fractal(vec3(pos.z+0, pos.z+0, pos.z+0),  abs(vec3(-0.12445855+0,0.84581727+0,-0.53769475+0)), fractal(vec3(pos.z+0, pos.z+0, pos.z+0),  (vec3(pos.x+ 0, pos.x+0, pos.x+0) / fractal(vec3(timevar+0, timevar+0, timevar+0),  vec3(pos.y+0, pos.y+0, pos.y+0), vec3(pos.z+0, pos.z+0, pos.z+0),  (vec3(-0.72968698+0,0.74443954+0,1.170503+0)* vec3(0.74443954+0,1.170503+0,-0.64745098+0)), true, true, vec3(pos.x+ 0, pos.x+0, pos.x+0), 1, 4, 4, 1, 1, 0, 3, 4, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0)), vec3(pos.y+0, pos.y+0, pos.y+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0), true, true, max(vec3(pos.y+0, pos.y+0, pos.y+0),  sin(vec3(-0.37429717+0,1.6037538+0,0.054174744+0))), 1, 4, 0, 0, 1, 1, 2, 1, 4, 0, 1, 1, 4, 3, 4, 1, 0, 0),  vec3(pos.y+0, pos.y+0, pos.y+0), true, true, vec3(pos.z+0, pos.z+0, pos.z+0), 4, 2, 2, 0, 1, 1, 1, 4, 0, 0, 1, 1, 2, 1, 4, 0, 1, 1)),  max(average(vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(timevar+0, timevar+0, timevar+0)),  transplantX(vec3(pos.y+0, pos.y+0, pos.y+0),  atan(vec3(0.76205647+0,0.91231072+0,-0.52285618+0)) + transplantX(transplantY(average(vec3(1.4001143+0,-0.24256553+0,-0.2464655+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0)),  vec3(-0.43984178+0,0.093410954+0,1.3026145+0)),  vec3(pos.y+0, pos.y+0, pos.y+0)))), abs(perlinNoiseVec3(vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(pos.x+ 0, pos.x+0, pos.x+0))),  min(transplantX((vec3(1.518353+0,-0.34136394+0,-0.29052988+0) / (vec3(0.76205647+0,0.91231072+0,-0.52285618+0) / vec3(timevar+0, timevar+0, timevar+0))),  vec3(pos.y+0, pos.y+0, pos.y+0)),  cos(transplantY(min(max(vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(timevar+0, timevar+0, timevar+0) - transplantX(sin(vec3(timevar+0, timevar+0, timevar+0)),  vec3(pos.x+ 0, pos.x+0, pos.x+0))),  cos(vec3(pos.z+0, pos.z+0, pos.z+0))),  transplantY(vec3(timevar+0, timevar+0, timevar+0) + min(fractal(vec3(pos.z+0, pos.z+0, pos.z+0),  min(vec3(pos.z+0, pos.z+0, pos.z+0),  vec3(timevar+0, timevar+0, timevar+0)), vec3(1.6489937+0,1.2269027+0,1.059414+0),  vec3(pos.y+0, pos.y+0, pos.y+0), true, true, vec3(pos.y+0, pos.y+0, pos.y+0), 1, 1, 3, 0, 0, 0, 2, 2, 4, 0, 0, 0, 0, 1, 4, 1, 0, 0),  transplantZ(vec3(pos.x+ 0, pos.x+0, pos.x+0),  vec3(pos.z+0, pos.z+0, pos.z+0))),  vec3(pos.x+ 0, pos.x+0, pos.x+0))))), true, true, fractal(vec3(timevar+0, timevar+0, timevar+0),  vec3(pos.y+0, pos.y+0, pos.y+0), vec3(pos.z+0, pos.z+0, pos.z+0),  (vec3(-0.72968698+0,0.74443954+0,1.170503+0)* vec3(0.74443954+0,1.170503+0,-0.64745098+0)), true, true, vec3(pos.x+ 0, pos.x+0, pos.x+0), 1, 4, 4, 1, 1, 0, 3, 4, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0), 4, 4, 0, 0, 0, 1, 0, 4, 3, 0, 1, 0, 1, 3, 2, 1, 1, 0)
```

</p>
</details>


### The L-System Tab

The L-System tab generates a tree according to the settings on the UI. The user can choose between 6 different L-Systems. Some of these options are 2D (these might show up sideways at first), and others are 3D. The user has the option to change the recursive depth, which determines the number of replacements in the L-system strings. Since the L-System trees generate the same thing each time, to add more variation, the user has the option to add length stochasticity and angle stochasticity. In length stochasticity, the length of each branch is shortened or lengthened by a random amount, and the angle is similarly randomly modified. The user also has the option to add leaves, which puts leaves at the end of each branch pointing downward and then in a random x/z direction to look more natural. As the user updates the settings, the tree will update and regenerate accordingly.

### The Gallery

The gallery tab randomly generates 5 trees with randomly selected L-Systems and recursive depths, and displays them in pots patterned with the current shaders on the shader evolution tab. 



## A little Devlog

#### [24/11/20 Update]

So we've uploaded a project plan in the readme_stuff folder. It's pretty detailed, though I'm sure we'll deviate from it when we start making this thing.



#### [29/11/20 Update]

So in the process of adding in some of the main courses's support code to this project (so we don't have to reinvent the wheel a thousand times), I ended up having to comment out  the dummy.cpp file in the glm/details. No idea what that does; hopefully that won't bite us in the butt later on. This is Anderson talking, by the way.



#### [06/12/20 Update]

Another update from Anderson! So the [paper](https://www.karlsims.com/papers/siggraph91.html) that the this project was inspired by mentions that they make use of iterated function systems (more about those [here](http://facstaff.susqu.edu/brakke/ifs/default.htm) and [here](http://soft.vub.ac.be/~tvcutsem/teaching/wpo/grafsys/ex4/les4.html). I would love to integrate them into this project, but making a proper IFS system would include the use of FBOs, and since some shader genotypes would use IFSs and some wouldn't (it's completely probabilistic, after all), that means there'd need to be conditional (and possibly nested) use of FBOs. Noooooo thank you.

So I'm gunning for a method for adding fractal like things using a methodology inspired by [this](https://www.mi.sanu.ac.rs/vismath/javier1/index.html) and [this](http://nuclear.mutantstargoat.com/articles/sdr_fract/).

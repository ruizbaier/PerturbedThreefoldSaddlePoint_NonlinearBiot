//lc=0.3; //coarse
lc = 0.12; //fine

Point(1) = {0, 0, 0, lc};
Point(2) = {1, 0, 0, lc};
Point(3) = {0.75, 0.5, 0, lc};
Point(4) = {0.5, 0.25, 0, lc};
Point(5) = {0.25,0.5, 0, lc};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5,1};

Line Loop(12) = {1, 2, 3, 4, 5};
Plane Surface(13) = {12};

//Transfinite Surface {13};

Physical Line(31) = {5,1};
Physical Line(32) = {2,3,4};

Physical Surface(41) = {13};

Mesh.ScalingFactor = 1.;
Mesh.Algorithm = 1;
Mesh.Format = 0;
Mesh.MshFileVersion = 2.2;

Mesh 2;
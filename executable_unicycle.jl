include("./src/utils.jl")
include("./src/dynamics.jl")
include("./src/constraint.jl")
include("./src/scaling.jl")
include("./src/trajopt.jl")

ix = 3
iu = 2
N = 20
dynamics = Unicycle()

function get_H_obs(rx,ry)
    return diagm([1/rx,1/ry])
end
c_list = []
H_list = []
c1 = [1,2]
H1 = get_H_obs(0.75,1.5)
push!(c_list,c1)
push!(H_list,H1)
c2 = [4,3]
H2 = get_H_obs(0.75,1.5)
push!(c_list,c2)
push!(H_list,H2)

vmax = 2
vmin = 0
wmax = 1
wmin = -1
Ac = [1 0;0 1;-1 0;0 -1];
bc = [vmax;wmax;-vmin;-wmin];
input_const = InputLinear(Ac,bc)
obstacle_const1 = Obstacle(H_list[1],c_list[1])
obstacle_const2 = Obstacle(H_list[2],c_list[2])
list_const = [input_const,obstacle_const1,obstacle_const2]

xi = [0;0;0];
xf = [5;5;0];

x0 = zeros(ix,N+1);
u0 = 0.1*ones(iu,N+1);
tf0 = 5.0;
dt0 = tf0/N*ones(N);
for i = 1:N+1
    x0[:,i] = (N-i+1)/N*xi+(i-1)/N*xf;
end

plt.figure()
ax = plt.subplot(111)
for (ce, H) in zip(c_list, H_list)
    rx = 1 / H[1, 1]  # Adjusted indexing for Julia (1-based indexing)
    ry = 1 / H[2, 2]  # Adjusted indexing for Julia
    circle1 = matplotlib[:patches][:Ellipse]((ce[1], ce[2]), width=rx*2, height=ry*2, color="tab:red", alpha=0.5, fill=true)
    ax[:add_patch](circle1)  # Using add_patch method to add the ellipse to the plot
end
ax.plot(x0[1,:],x0[2,:],"--",color="tab:blue")
ax.plot(xi[1],xi[2],"o",color="tab:green")
ax.plot(xf[1],xf[2],"o",color="tab:green")
ax.grid(true)
ax[:axis]("equal")
gcf()

xmin = [0;0;0];
xmax = [5;5;pi];
umin = [0;0];
umax = [2;1];
min_dt = 0.1;
max_dt = 0.5;
scaler = Scaling(xmin, xmax, umin, umax, tf0, min_dt,max_dt)
@assert max_dt * N >= tf0 

max_iter = 30;
w_tf = 0.0;
w_c = 1e-1;
w_rate = 1e-3;
w_param = 0.0;
w_vc = 1e2;
w_tr::Float64 = 5*1e-1;
tol_vc = 1e-6;
tol_tr = 1e-4;
tol_dyn = 1e-1;
tr_norm = "quad";
verbosity = true;

ptr = PTR(N,tf0,max_iter,dynamics,list_const,scaler,
    w_tf,w_c,w_rate,w_param,w_vc,w_tr,
    tol_vc,tol_tr,tol_dyn,
    tr_norm,verbosity)

run(ptr,x0,u0,dt0,xi,xf)

plt.figure()
ax = plt.subplot(111)
for (ce, H) in zip(c_list, H_list)
    rx = 1 / H[1, 1]  # Adjusted indexing for Julia (1-based indexing)
    ry = 1 / H[2, 2]  # Adjusted indexing for Julia
    circle1 = matplotlib[:patches][:Ellipse]((ce[1], ce[2]), width=rx*2, height=ry*2, color="tab:red", alpha=0.5, fill=true)
    ax[:add_patch](circle1)  # Using add_patch method to add the ellipse to the plot
end
ax.plot(ptr.solution.x[1,:],ptr.solution.x[2,:],"o",color="tab:blue")
ax.plot(ptr.solution.xprop[1,:],ptr.solution.xprop[2,:],"-",color="tab:blue")
ax.grid(true)
ax[:axis]("equal")
# gcf()

plt.figure(figsize=(10,3))
plt.subplot(121)
plt.plot(ptr.solution.t,ptr.solution.u[1,:])
plt.plot(ptr.solution.t,ptr.solution.t*0 .+ vmax,"--",color="tab:red")
plt.plot(ptr.solution.t,ptr.solution.t*0 .+ vmin,"--",color="tab:red")
plt.ylim([-0.1,2.1])
plt.grid(true)
plt.subplot(122)
plt.plot(ptr.solution.t,ptr.solution.u[2,:])
plt.plot(ptr.solution.t,ptr.solution.t*0 .+ wmax,"--",color="tab:red")
plt.plot(ptr.solution.t,ptr.solution.t*0 .+ wmin,"--",color="tab:red")
plt.ylim([-2,2])
plt.grid(true)
plt.show()
# gcf()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

G = 1
dt = 0.05
N = 100

t = np.linspace(0,100,500)

plt.style.use('dark_background')
fig, axis = plt.subplots(figsize=(14, 8))
axis.set_xlim(-100,100)
axis.set_ylim(-100,100)
axis.set_xticks([])
axis.set_yticks([])
axis.set_title('N-Body Simulation (2 Dimensions)')
#plt.grid()

planetobj = {}
for i in range(N):
    new_planet_plot, = axis.plot([], [], 'o', markersize=15, color='white')
    planetobj[i] = new_planet_plot

pos = np.zeros((N,2))
masses = np.full((N), 200.0)
pos_0 = np.full((N,2), 0.0)

l = 0 
def spawn(event):
    global l, planetdata, pos

    if event.inaxes:
        x = event.xdata
        y = event.ydata
        planetobj[l].set_data([x],[y])
        pos[l] = [x,y]
        pos_0[l] = [x,y]
        l += 1
        fig.canvas.draw_idle()

def get_forces(pos, mass, G=4, epsilon=4):
    diff = pos[None, :, :] - pos[:, None, :] #N, N, 2 Force Tensor
    r = np.linalg.norm(diff, axis=2) #N,N
    r[r < epsilon] = epsilon 
    m_product = mass[:, None] * mass[None, :]
    force_scalar = (G * m_product) / (r**3)
    force_matrix = force_scalar[:, :, None] * diff
    net_force = np.sum(force_matrix, axis=1)
    
    return net_force


def update(frame):
    global l, pos, pos_0
    if l > 0:
        active_pos = pos[:l]
        active_mass = masses[:l]
        active_pos_0 = pos_0[:l]

        forces = get_forces(active_pos, active_mass)
        acc = forces / active_mass[:, None]

        pos_current = active_pos.copy()

        active_pos[:] = 2 * active_pos - active_pos_0 + acc * dt**2

        active_pos_0[:] = pos_current

    for i in range(l):
        planetobj[i].set_data([pos[i, 0]], [pos[i, 1]])

    return list(planetobj.values())

world = FuncAnimation(
    fig = fig,
    func = update,
    frames = len(t),
    interval = 25,
)
    
cid = fig.canvas.mpl_connect('button_press_event', spawn)

plt.show()

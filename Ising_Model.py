import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class IsingModel:
    def __init__(self, size=50, temp=2.27, interaction_strength=1.0):

        self.size = size
        self.temp = temp
        self.J = interaction_strength
        
        self.grid = np.random.choice([-1, 1], size=(size, size))
        
    def _get_delta_energy(self, i, j):

        spin = self.grid[i, j]
        
        neighbors = (
            self.grid[(i + 1) % self.size, j] +
            self.grid[(i - 1) % self.size, j] +
            self.grid[i, (j + 1) % self.size] +
            self.grid[i, (j - 1) % self.size]
        )
        
        return 2 * self.J * spin * neighbors

    def step(self):

        for i in range(self.size * self.size):

            i, j = np.random.randint(0, self.size, 2)
            
            dE = self._get_delta_energy(i, j)
            

            if dE < 0 or np.random.rand() < np.exp(-dE / self.temp):
                self.grid[i, j] *= -1


def animate_ising():
    #simulation characteristics
    N = 100            
    T = 2.27           
    
    #model
    model = IsingModel(size=N, temp=T)

    fig, ax = plt.subplots()
    ax.set_title(f"2D Ising Model (T = {T})", fontdict={'fontname': 'Times New Roman', 'fontsize': 20})
    

    img = ax.imshow(model.grid, cmap='coolwarm', interpolation='nearest')
    plt.axis('off') #Hides Axes (change if unwanted)

    def update(frame):

        model.step()
        
        #Update Image data (is this the fastest way?)
        img.set_data(model.grid)
        return img,

    #Animate Plot
    ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)
    
    plt.show()

if __name__ == "__main__":
    animate_ising()

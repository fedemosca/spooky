''' 2D Shallow Water Equations '''

import numpy as np
import os

from .._backend import xnp, index_update, apply_jit
from .pseudospectral import PseudoSpectral
from .. import pseudo as ps


class SWHD_2D(PseudoSpectral):
    ''' 2D Shallow Water Equations
        ut + u ux + v uy + g hx = 0
        vt + u vx + v vy + g hy = 0
        ht + ( u(h-hb) )x + ( v(h-hb) )y = 0
    where u,v,h are velocity and height fields,
    and hb is bottom topography condition.
    '''

    num_fields = 3
    dim_fields = 2
    def __init__(self,
                 grid: ps.Grid2D, pm, rkord=2):
        super().__init__(grid, rkord)
        self.pm = pm
        self.make_data = pm.make_data
        self.noise = pm.noise
        self.uum_noise_std = pm.uum_noise_std
        self.vvm_noise_std = pm.vvm_noise_std
        self.hhm_noise_std = pm.hhm_noise_std
        self.iit = pm.iit
        self.iit0 = pm.iit0
        self.iitN = pm.iitN
        self.total_iterations = self.iitN - self.iit0 - 1
        self.total_steps =  int(self.pm.T/self.pm.dt)+1 # total time steps, since RK runs for Nt+1 steps
        self.step = 0 # current step for adjoint solver
        self.data_path = pm.data_path
        self.hb_path = pm.hb_path
        self.uus = None
        self.vvs = None
        self.hhs = None
        self.uus_noise = None
        self.vvs_noise = None
        self.hhs_noise = None
        self.hb = None
        self.true_hb = None

    def update_true_hb(self):
        self.true_hb = np.load(f'{self.data_path}/hb.npy')

    def update_hb(self, hb):
        self.hb = hb

    def rkstep(self, fields, prev, oo, dt):
        ''' Passes hb as a traced argument.

        Under jit `self` is a static argument, so any attribute read inside the compiled
        function is baked in at trace time and never refreshed: a loop calling update_hb()
        between evolve() calls would silently keep the first topography.
        '''
        return self._rkstep(fields, prev, oo, dt, self.hb)

    @apply_jit
    def _rkstep(self, fields, prev, oo, dt, hb):
        # Unpack
        fu  = fields[0]
        fup = prev[0]

        fv  = fields[1]
        fvp = prev[1]

        fh  = fields[2]
        fhp = prev[2]

        # Non-linear term
        uu = self.grid.inverse(fu)
        vv = self.grid.inverse(fv)
        hh = self.grid.inverse(fh)

        ux = self.grid.inverse(self.grid.deriv(fu, self.grid.kx))
        uy = self.grid.inverse(self.grid.deriv(fu, self.grid.ky))

        vx = self.grid.inverse(self.grid.deriv(fv, self.grid.kx))
        vy = self.grid.inverse(self.grid.deriv(fv, self.grid.ky))

        fhx = self.grid.deriv(fh, self.grid.kx) # i k_i fh_i
        fhy = self.grid.deriv(fh, self.grid.ky) # i k_j fh_j

        fu_grad_u = self.grid.forward(uu*ux + vv*uy)
        fv_grad_v = self.grid.forward(uu*vx + vv*vy)

        fu_h_hb_x = self.grid.deriv(self.grid.forward(uu*(hh-hb)), self.grid.kx)
        fv_h_hb_y = self.grid.deriv(self.grid.forward(vv*(hh-hb)), self.grid.ky)

        fu = fup + (dt/oo) * (-fu_grad_u - self.pm.g*fhx)
        fv = fvp + (dt/oo) * (-fv_grad_v - self.pm.g*fhy)
        fh = fhp + (dt/oo) * (-fu_h_hb_x - fv_h_hb_y)

        # de-aliasing
        fu = index_update(fu, self.grid.zero_mode, 0.0)
        fu = index_update(fu, self.grid.dealias_modes, 0.0)
        fv = index_update(fv, self.grid.zero_mode, 0.0)
        fv = index_update(fv, self.grid.dealias_modes, 0.0)
        # zero mode for h is not null
        fh = index_update(fh, self.grid.dealias_modes, 0.0)

        return [fu, fv, fh]

    def save_to_ram(self, storage, new_data, step, total_steps, dtype=np.float64):
        """
        Saves new data to an in-memory storage array.

        Args:
            storage (np.ndarray or None): The in-memory storage array. Pass `None` on the first call to initialize it.
            new_data (np.ndarray): Data to be saved at the current iteration.
            step (int): Current iteration (used to index the storage array).
            total_steps (int): Total number of iterations to preallocate space for.
            dtype (type): Data type of the saved array (default: np.float64).

        Returns:
            np.ndarray: Updated storage array.
        """
        if step == 0:
            # Initialize the in-memory storage array with preallocated space
            storage = np.zeros((total_steps,) + new_data.shape, dtype=dtype)
        # Save the new data into the corresponding slot
        storage[step] = new_data

        return storage

# Save hb and dg arays in file
    def save_memmap(self, filename, new_data, step, total_steps, dtype=np.float64):
        """
        Saves new data to an existing or new preallocated memory-mapped .npy file.

        Args:
            filename (str): Path to the memory-mapped .npy file.
            new_data (np.ndarray): Data to be saved at the current iteration.
            step (int): Current iteration (used to index the memory-mapped file).
            total_steps (int): Total number of iterations to preallocate space for.
            dtype (type): Data type of the saved array (default: np.float64).
        """
        if step == 0:
            # Create a new memory-mapped file with preallocated space for all iterations
            if os.path.exists(filename):
                os.remove(filename)
            # Shape includes space for total_iterations along the first axis
            shape = (total_steps,) + new_data.shape  # Preallocate for total iterations
            fp = np.lib.format.open_memmap(filename, mode='w+', dtype=dtype, shape=shape)
        else:
            # Load the existing memory-mapped file (no need to resize anymore)
            fp = np.load(filename, mmap_mode='r+')

        # Write new data into the current iteration slot
        fp[step] = new_data
        del fp  # Force the file to flush and close

    def add_noise(self, field, mean=0.0, std=1.0):
        noise = np.random.normal(loc=mean, scale=std, size=field.shape)
        return field + noise

    def outs(self, fields, step, opath):
        ostep = self.pm.ostep
        # Steps run 0..total_steps-1, so the last output slot is (total_steps-1)//ostep
        n_outs = (self.total_steps - 1)//ostep + 1
        out_step = step//ostep

        uu = self.grid.inverse(fields[0])
        self.uus = self.save_to_ram(self.uus, uu, out_step, n_outs, dtype=np.float64)

        vv = self.grid.inverse(fields[1])
        self.vvs = self.save_to_ram(self.vvs, vv, out_step, n_outs, dtype=np.float64)

        hh = self.grid.inverse(fields[2])
        self.hhs = self.save_to_ram(self.hhs, hh, out_step, n_outs, dtype=np.float64)

        if self.make_data and (step == self.total_steps-1):

            np.save(f'{self.pm.out_path}/uums', self.uus)
            np.save(f'{self.pm.out_path}/vvms', self.vvs)
            np.save(f'{self.pm.out_path}/hhms', self.hhs)

    def balance(self, fields, step, bpath):
        eng = self.grid.energy(fields)
        bal = [f'{self.pm.dt*step:.4e}', f'{eng:.6e}']
        with open(f'{self.pm.out_path}/balance.dat', 'a') as output:
            print(*bal, file=output)

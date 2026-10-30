# String method based on the user-supplied parametric implementation.
# Original author: Akash Pallath. Original implementation licensed under MIT.
# Reference: E, Ren & Vanden-Eijnden, JCP 126, 164103 (2007).
"""Parametric string relaxation with uniform-grid interpolation and support checks."""
import heapq
import warnings
import numpy as np
from scipy.interpolate import make_interp_spline, RegularGridInterpolator
from scipy.ndimage import map_coordinates, distance_transform_edt
from tqdm import tqdm


def resample(path,npts,scales=(1.,1.)):
    p=np.asarray(path,float)
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0)/scales,axis=1))]
    keep=np.r_[True,np.diff(arc)>1e-14]
    p,arc=p[keep],arc[keep]
    if len(p)<2:
        raise ValueError('The string collapsed to one point.')
    return np.column_stack([np.interp(np.linspace(0,arc[-1],npts),arc,p[:,j]) for j in range(2)])


class String2D:
    """Same main interface as stringmethod_parametric.String2D.

    V is (ny,nx) for indexing='xy', (nx,ny) for 'ij'. Finite grid support is
    enforced: missing FES cells are not filled into an artificial reaction path.
    scales specify the CV metric; default is the original raw Euclidean metric.
    Convergence refers to numerical string motion, not sampling or a committor.
    """
    def __init__(self,x,y,V,indexing='xy',scales=(1.,1.)):
        self.x,self.y=np.asarray(x,float),np.asarray(y,float)
        self.scales=np.asarray(scales,float)
        self.indexing=indexing
        if indexing not in ('ij','xy') or self.scales.shape!=(2,) or np.any(self.scales<=0):
            raise ValueError("Use indexing='xy'/'ij' and two positive metric scales.")
        for a in (self.x,self.y):
            if len(a)<3 or np.any(np.diff(a)<=0) or not np.allclose(np.diff(a),a[1]-a[0]):
                raise ValueError('String interpolation requires increasing, uniform axes (>=3 points).')
        self.V=np.asarray(V,float).copy()
        z=self.V.T if indexing=='xy' else self.V
        if z.shape!=(len(x),len(y)):
            raise ValueError('V shape and indexing disagree.')
        self._z=z;self._valid=np.isfinite(z)
        if not self._valid.any():
            raise ValueError('No finite surface points.')
        # Fill for finite-difference gradients and to avoid 0*NaN in interpolation.
        # Every string segment is still checked against the ORIGINAL support mask.
        filled=z.copy()
        if not self._valid.all():
            near=distance_transform_edt(~self._valid,return_distances=False,return_indices=True)
            filled[~self._valid]=z[tuple(near[:,~self._valid])]
        gx,gy=np.gradient(filled,self.x,self.y)
        self.gradX_array,self.gradY_array=gx,gy
        self.gradX,self.gradY=(gx.T,gy.T) if indexing=='xy' else (gx,gy)
        self.X,self.Y=np.meshgrid(self.x,self.y,indexing=indexing)
        self.grid=np.column_stack((self.X.ravel(),self.Y.ravel()))
        self._energy=RegularGridInterpolator((self.x,self.y),filled,bounds_error=True)
        self.string_traj=[];self.mep=None;self.converged=False;self.convergence_history=[]
        self.normal_force_history=[]

    def to_grid_coords(self,points):
        return ((np.asarray(points)-[self.x[0],self.y[0]])/
                [self.x[1]-self.x[0],self.y[1]-self.y[0]]).T

    def _supported(self,points):
        p=np.asarray(points)
        inside=np.all((p>=[self.x[0],self.y[0]])&(p<=[self.x[-1],self.y[-1]]))
        if not inside:
            return False
        coords = self.to_grid_coords(p).T
        # Also test BETWEEN images: sparse string images must not jump over holes.
        if len(coords) > 1:
            n = np.maximum(1, np.ceil(2*np.max(abs(np.diff(coords,axis=0)),axis=1)).astype(int))
            segments = [np.linspace(a,b,k+1) for a,b,k in zip(coords[:-1],coords[1:],n)]
            coords = np.concatenate(segments)
        coverage=map_coordinates(self._valid.astype(float),coords.T,order=1,mode='constant')
        return bool(np.all(coverage>1.-1e-10))

    def prepare_interpolation(self):
        """Compatibility method; arrays are already prepared at construction."""
        return self

    def interpolate_gradients_map(self,points):
        coords=self.to_grid_coords(points)
        return tuple(map_coordinates(a,coords,order=1,mode='nearest')
                     for a in (self.gradX_array,self.gradY_array))

    def step_euler_new(self,string,dt,flexible=True):
        g=np.column_stack(self.interpolate_gradients_map(string))*self.scales
        if not flexible:
            g[[0,-1]]=0.
        h=np.linalg.norm(g,axis=1).max()
        return string.copy() if h<1e-14 else string-dt*g*self.scales/h

    step_euler=step_euler_new

    def compute_mep(self,begin,end,mid=None,spline_order=2,npts=100,
                    integrator='forward_euler',dt=.02,tol=1e-5,maxsteps=1000,
                    traj_every=50,flexible=True,initial_path=None,progress=True):
        if integrator!='forward_euler' or dt<=0 or maxsteps<1 or npts<3 or traj_every<1:
            raise ValueError('Use forward_euler, dt>0, maxsteps>=1, npts>=3, traj_every>=1.')
        self.converged=False;self.convergence_history=[];self.normal_force_history=[]
        if initial_path is None:
            control=np.asarray([begin]+([] if mid is None else list(mid))+[end],float)
            lengths=np.linalg.norm(np.diff(control,axis=0)/self.scales,axis=1)
            if np.any(lengths==0):
                raise ValueError('Consecutive control points must be distinct.')
            t=np.r_[0,np.cumsum(lengths)];t/=t[-1]
            k=min(spline_order,len(control)-1)
            string=make_interp_spline(t,control,k=k)(np.linspace(0,1,npts))
        else:
            string=resample(initial_path,npts,self.scales)
            if not np.allclose(string[[0,-1]],[begin,end]):
                raise ValueError('initial_path endpoints must match begin and end.')
        if not self._supported(string):
            raise ValueError('Initial string crosses unsupported/outside FES cells. '
                             'Use a supported midpoint or minimax_path initializer.')
        self.string_traj=[string.copy()]
        for step in tqdm(range(maxsteps),disable=not progress,desc='String'):
            old=string.copy()
            # Backtracking protects support, but is not used to declare convergence.
            trial_dt=dt
            for attempt in range(12):
                trial=resample(self.step_euler_new(old,trial_dt,flexible),npts,self.scales)
                if self._supported(trial):
                    break
                trial_dt*=.5
            else:
                warnings.warn('String reached unsupported grid cells; stopping without convergence.')
                break
            string=trial
            rms=float(np.sqrt(np.mean(((string-old)/self.scales)**2)))
            self.convergence_history.append(rms)
            grad=np.column_stack(self.interpolate_gradients_map(string))*self.scales
            tangent=np.gradient(string/self.scales,axis=0)
            tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-15)
            normal=grad-(grad*tangent).sum(axis=1)[:,None]*tangent
            if flexible:
                normal[[0,-1]]=grad[[0,-1]]
            residual = normal if flexible else normal[1:-1]
            self.normal_force_history.append(float(np.sqrt(np.mean(residual**2))))
            if (step+1)%traj_every==0:
                self.string_traj.append(string.copy())
            if rms<tol and attempt==0:
                self.converged=True;break
        self.mep=string
        self.string_traj.append(string.copy())
        if not self.converged:
            warnings.warn('MEP not converged. Inspect convergence_history; reduce dt/increase maxsteps. '
                          'Do not report a barrier from this path as converged.')
        return self.mep

    def get_mep_energy_profile(self):
        if self.mep is None:
            raise ValueError('Run compute_mep first.')
        return self.mep,self._energy(self.mep)

    def plot_V(self,clip_min=None,clip_max=None,levels=20,cmap=None,dpi=150):
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(dpi=dpi)
        v=np.ma.masked_invalid(self._z.T)
        if clip_min is not None or clip_max is not None:
            v=np.clip(v,clip_min,clip_max)
        cs=ax.contourf(self.x,self.y,v,levels=levels,cmap=cmap)
        return fig,ax,fig.colorbar(cs,ax=ax)

    def plot_mep(self,**kwargs):
        fig,ax,bar=self.plot_V(**kwargs)
        ax.plot(*self.mep.T);ax.scatter(*self.mep[[0,-1]].T)
        return fig,ax,bar

    def plot_mep_energy_profile(self,dpi=150):
        import matplotlib.pyplot as plt
        path,e=self.get_mep_energy_profile()
        arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0)/self.scales,axis=1))]
        fig,ax=plt.subplots(dpi=dpi);ax.plot(arc/arc[-1],e)
        return fig,ax

    def plot_string_evolution(self,**kwargs):
        fig,ax,bar=self.plot_V(**kwargs)
        for s in self.string_traj:
            ax.plot(*s.T,alpha=.4)
        return fig,ax,bar


def minimax_path(x,y,free_energy,begin,end,*,scales=(1.,1.)):
    """8-neighbor minimum-maximum-energy grid route; an INITIALIZER, not a MFEP.

    Path length is a tie-breaking heuristic, not a globally optimal secondary
    objective. Unsupported cells and diagonal corner-cutting are
    forbidden. Endpoints snap to the nearest grid points; returned path exposes them.
    """
    x,y,z=np.asarray(x),np.asarray(y),np.asarray(free_energy)
    shape=(len(x),len(y));valid=np.isfinite(z)
    if z.shape!=shape:
        raise ValueError("free_energy must use indexing='ij'.")
    start=(int(np.argmin(abs(x-begin[0]))),int(np.argmin(abs(y-begin[1]))))
    goal=(int(np.argmin(abs(x-end[0]))),int(np.argmin(abs(y-end[1]))))
    if not valid[start] or not valid[goal]:
        raise ValueError('Endpoints must lie on supported cells.')
    costs={start:(float(z[start]),0.)};previous={};queue=[(*costs[start],start)]
    moves=[(i,j) for i in (-1,0,1) for j in (-1,0,1) if i or j]
    while queue:
        height,length,node=heapq.heappop(queue)
        if costs[node]!=(height,length):
            continue
        if node==goal:
            break
        a,b=node
        for da,db in moves:
            c,d=a+da,b+db
            if not(0<=c<shape[0] and 0<=d<shape[1]) or not valid[c,d]:
                continue
            if da and db and not(valid[c,b] and valid[a,d]):
                continue
            step=np.hypot((x[c]-x[a])/scales[0],(y[d]-y[b])/scales[1])
            cost=(max(height,float(z[c,d])),length+step)
            if cost<costs.get((c,d),(np.inf,np.inf)):
                costs[(c,d)]=cost;previous[(c,d)]=node;heapq.heappush(queue,(*cost,(c,d)))
    if goal not in costs:
        raise ValueError('No connected supported grid route between endpoints.')
    nodes=[goal]
    while nodes[-1]!=start:
        nodes.append(previous[nodes[-1]])
    return np.array([(x[i],y[j]) for i,j in nodes[::-1]])

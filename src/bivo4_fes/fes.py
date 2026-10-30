"""Stable 1D/2D weighted KDE, fast binned KDE, and portable grid I/O.

Grid arrays always use indexing='ij': F[i,j] is F(x[i],y[j]).
The local backend evaluates Gaussians at grid points (no coordinate binning).
It truncates each coordinate at cutoff*sigma, unlike the exact reference backend.
"""
from dataclasses import dataclass
import json
import warnings
import numpy as np
from numba import njit
from scipy.special import logsumexp
from scipy.ndimage import convolve1d
from scipy.interpolate import RegularGridInterpolator
from .reweight import kbt, weight_diagnostics
from .separable import gaussian_moments


@njit(cache=True)
def _deposit(samples,lw,x,y,bw,cutoff):
    logp=np.full((len(x),len(y)),-np.inf)
    logp2=logp.copy(); count=np.zeros(logp.shape,np.int64)
    hx=x[1]-x[0]; hy=y[1]-y[0]
    for i in range(len(samples)):
        if not np.isfinite(lw[i]):
            continue
        lo=max(0,int(np.ceil((samples[i,0]-cutoff*bw[0]-x[0])/hx)))
        hi=min(len(x),int(np.floor((samples[i,0]+cutoff*bw[0]-x[0])/hx))+1)
        ly=max(0,int(np.ceil((samples[i,1]-cutoff*bw[1]-y[0])/hy)))
        uy=min(len(y),int(np.floor((samples[i,1]+cutoff*bw[1]-y[0])/hy))+1)
        for a in range(lo,hi):
            z=lw[i]-.5*((x[a]-samples[i,0])/bw[0])**2
            for b in range(ly,uy):
                q=z-.5*((y[b]-samples[i,1])/bw[1])**2
                logp[a,b]=np.logaddexp(logp[a,b],q)
                logp2[a,b]=np.logaddexp(logp2[a,b],2*q)
                count[a,b]+=1
    return logp,logp2,count


@njit(cache=True)
def _deposit1(samples,lw,x,bw,cutoff):
    logp=np.full(len(x),-np.inf); logp2=logp.copy(); count=np.zeros(len(x),np.int64)
    hx=x[1]-x[0]
    for i in range(len(samples)):
        if not np.isfinite(lw[i]):
            continue
        lo=max(0,int(np.ceil((samples[i]-cutoff*bw-x[0])/hx)))
        hi=min(len(x),int(np.floor((samples[i]+cutoff*bw-x[0])/hx))+1)
        for a in range(lo,hi):
            q=lw[i]-.5*((x[a]-samples[i])/bw)**2
            logp[a]=np.logaddexp(logp[a],q)
            logp2[a]=np.logaddexp(logp2[a],2*q)
            count[a]+=1
    return logp,logp2,count


def _exact(samples,lw,axes,bw):
    """Slow all-sample reference for validation; chunked to bound memory."""
    grid=np.column_stack([a.ravel() for a in np.meshgrid(*axes,indexing='ij')])
    lp=np.full(len(grid),-np.inf); lp2=lp.copy()
    for start in range(0,len(grid),64):
        points=grid[start:start+64]
        p=np.full(len(points),-np.inf); p2=p.copy()
        for j in range(0,len(samples),4096):
            z=lw[j:j+4096,None]-.5*np.sum(((samples[j:j+4096,None,:]-points[None,:,:])/bw)**2,axis=2)
            p=np.logaddexp(p,logsumexp(z,axis=0))
            p2=np.logaddexp(p2,logsumexp(2*z,axis=0))
        lp[start:start+64]=p;lp2[start:start+64]=p2
    shape=tuple(map(len,axes))
    return lp.reshape(shape),lp2.reshape(shape),np.full(shape,len(samples))


def _binned(samples,lw,axes,bw,cutoff):
    """Positive convolution, not FFT (avoids FFT noise floors in rare basins).

    Binning is an approximation. Require at least two grid intervals per sigma.
    Padding includes nearby samples OUTSIDE the requested plotting bounds.
    """
    steps=np.array([a[1]-a[0] for a in axes]); sigmas=bw/steps
    if np.any(sigmas<2.):
        raise ValueError("Binned KDE needs spacing <= bandwidth/2. Refine bins or use method='local'.")
    radius=np.ceil(cutoff*sigmas).astype(int)
    edges=[a[0]+(np.arange(-r,len(a)+r+1)-.5)*h for a,r,h in zip(axes,radius,steps)]
    shift=np.max(lw); w=np.exp(lw-shift)
    mass=np.histogramdd(samples,bins=edges,weights=w)[0]
    second=np.histogramdd(samples,bins=edges,weights=w*w)[0]
    count=np.histogramdd(samples,bins=edges)[0]
    for dim,(s,r) in enumerate(zip(sigmas,radius)):
        kernel=np.exp(-.5*(np.arange(-r,r+1)/s)**2)  # unnormalized, as local KDE
        mass=convolve1d(mass,kernel,axis=dim,mode='constant')
        second=convolve1d(second,kernel*kernel,axis=dim,mode='constant')
        count=convolve1d(count,np.ones(len(kernel)),axis=dim,mode='constant')
    sl=tuple(slice(r,r+len(a)) for a,r in zip(axes,radius))
    with np.errstate(divide='ignore'):
        return np.log(mass[sl])+shift,np.log(second[sl])+2*shift,count[sl].astype(int)


@dataclass
class FES:
    axes: tuple
    free_energy: np.ndarray
    log_density: np.ndarray
    local_ess: np.ndarray
    count: np.ndarray
    metadata: dict

    @property
    def unit(self):
        return self.metadata['unit']

    def plot(self, *, ax=None, vmax=None, min_ess=0, **kwargs):
        import matplotlib.pyplot as plt
        if ax is None:
            _,ax=plt.subplots()
        f=self.free_energy.copy()
        f[(self.local_ess<min_ess)|~np.isfinite(f)]=np.nan
        if len(self.axes)==1:
            ax.plot(self.axes[0],f,**kwargs); ax.set_ylabel(f'Free energy ({self.unit})')
        else:
            if vmax is not None:
                f=np.minimum(f,vmax)
            artist=ax.contourf(*self.axes,f.T,**kwargs)
            ax.figure.colorbar(artist,ax=ax,label=f'Free energy ({self.unit})')
        return ax

    def profile(self,path):
        """Bilinear interpolation on the original grid, not clipped plot values."""
        valid = np.isfinite(self.free_energy)
        energy = RegularGridInterpolator(self.axes,np.where(valid,self.free_energy,0.),bounds_error=True)(path)
        coverage = RegularGridInterpolator(self.axes,valid.astype(float),bounds_error=True)(path)
        return np.where(coverage >= 1.-1e-12,energy,np.nan)

    def save(self,path):
        np.savez_compressed(path,**{f'axis{i}':a for i,a in enumerate(self.axes)},
                            free_energy=self.free_energy,log_density=self.log_density,
                            local_ess=self.local_ess,count=self.count,metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as f:
            meta=json.loads(str(f['metadata']))
            axes=tuple(f[f'axis{i}'] for i in range(f['free_energy'].ndim))
            return cls(axes,f['free_energy'],f['log_density'],f['local_ess'],f['count'],meta)

    def to_plumed(self,path,names=None):
        names=names or [f'cv{i+1}' for i in range(len(self.axes))]
        coords=np.meshgrid(*self.axes,indexing='ij')
        # PLUMED's first coordinate varies fastest.
        columns=[a.ravel(order='F') for a in coords]+[self.free_energy.ravel(order='F')]
        header='#! FIELDS '+' '.join(names+['free'])+'\n'
        header+=f'#! SET energy_unit {self.unit}\n'
        for name,a in zip(names,self.axes):
            header+=f'#! SET min_{name} {a[0]}\n#! SET max_{name} {a[-1]}\n'
            header+=f'#! SET nbins_{name} {len(a)}\n#! SET periodic_{name} false\n'
        with open(path,'w') as f:
            f.write(header);np.savetxt(f,np.column_stack(columns),fmt='%.10g')


def compute_fes(samples,logweights,*,bounds,bins=101,bandwidth=0.03,
                temperature=350.,unit='eV',method='local',cutoff=12.,log_normalizer=None):
    """Weighted 1D/2D FES on a uniform grid (bins = number of GRID POINTS).

    samples: (n,) or (n,dim); logweights: one original OPES log-weight per frame.
    NaN CVs are excluded together with their weights. Default normalization is
    conditional on valid samples (e.g. direct-connected configurations). Supply
    log_normalizer=logsumexp(ALL retained frame weights) for a joint density.
    free_energy is minimum-shifted for plotting; log_density preserves normalization.
    Channels that overlap are not mutually exclusive kinetic mechanisms.
    method='separable' uses exact 2D matrix products with no kernel truncation
    or coordinate binning (1D falls back to the exact reference).
    """
    q=np.asarray(samples,dtype=float)
    if q.ndim==1:
        q=q[:,None]
    lw=np.asarray(logweights,dtype=float)
    if q.ndim!=2 or q.shape[1] not in (1,2) or lw.shape!=(len(q),):
        raise ValueError('Provide n samples in 1D/2D and one log-weight per sample.')
    if np.isnan(lw).any() or np.isposinf(lw).any():
        raise ValueError("Log-weights cannot contain NaN or +inf.")
    good=np.all(np.isfinite(q),axis=1)&np.isfinite(lw)
    q,lw=q[good],lw[good]
    if not len(q):
        raise ValueError('No finite CV/weight pairs in this channel.')
    dim=q.shape[1];bw=np.broadcast_to(np.asarray(bandwidth,dtype=float),(dim,)).copy()
    bins=np.broadcast_to(np.asarray(bins,dtype=int),(dim,))
    bounds=np.asarray(bounds,dtype=float).reshape(dim,2)
    if (not np.all(np.isfinite(bw)) or not np.all(np.isfinite(bounds)) or cutoff<=0
            or np.any(bw<=0) or np.any(bins<2) or np.any(bounds[:,1]<=bounds[:,0])):
        raise ValueError('Positive bandwidths, >=2 grid points and increasing bounds required.')
    axes=tuple(np.linspace(lo,hi,n) for (lo,hi),n in zip(bounds,bins))
    if np.any(np.array([a[1]-a[0] for a in axes])>bw):
        warnings.warn('Grid spacing exceeds KDE bandwidth. Values are evaluated correctly, '
                      'but the surface/MEP may be under-resolved; test a finer grid.',UserWarning)
    if method=='local':
        if dim==1:
            lp,lp2,count=_deposit1(q[:,0],lw,axes[0],bw[0],cutoff)
        else:
            lp,lp2,count=_deposit(q,lw,*axes,bw,cutoff)
    elif method=='exact':
        lp,lp2,count=_exact(q,lw,axes,bw)
    elif method=='separable':
        lp,lp2,count=gaussian_moments(q,lw,axes,bw) if dim==2 else _exact(q,lw,axes,bw)
    elif method=='binned':
        lp,lp2,count=_binned(q,lw,axes,bw,cutoff)
    else:
        raise ValueError('method must be local, separable, binned, or exact.')
    norm=logsumexp(lw) if log_normalizer is None else float(log_normalizer)
    kernel_norm=np.sum(np.log(np.sqrt(2*np.pi)*bw))
    logdensity=lp-norm-kernel_norm
    if not np.isfinite(logdensity).any():
        raise ValueError('No kernel supports this grid; revise bounds or inspect the channel.')
    free=-kbt(temperature,unit)*(logdensity-np.max(logdensity))
    ess=np.zeros_like(lp);valid=np.isfinite(lp)
    ess[valid]=np.exp(2*lp[valid]-lp2[valid])
    meta=dict(unit=unit,temperature=temperature,method=method,bandwidth=bw.tolist(),cutoff=cutoff,
              normalization='selected' if log_normalizer is None else 'external',
              input_frames=len(good),selected_frames=len(q),log_normalizer=float(norm),
              **weight_diagnostics(lw))
    # Conservative absolute density-error bound for omitted tails of local KDE.
    # It need not be tight; compare it to the density where the path lies.
    meta['log_tail_density_bound']=(float(logsumexp(lw)-norm-kernel_norm-.5*cutoff**2)
                                    if method=='local' else None)
    return FES(axes,free,logdensity,ess,count,meta)


def read_fes(path,*,energy_unit=None,output_unit='eV',dimension=2):
    """Read a complete rectangular text grid; infer order, never assume 101x101."""
    if energy_unit is None:
        with open(path) as stream:
            for line in stream:
                if not line.startswith('#'):
                    break
                if line.startswith('#! SET energy_unit'):
                    energy_unit = line.split()[-1]
        if energy_unit is None:
            raise ValueError("No energy_unit in the grid header; specify energy_unit explicitly.")
    a=np.loadtxt(path,comments='#',ndmin=2)
    axes=tuple(np.unique(a[:,i]) for i in range(dimension))
    shape=tuple(map(len,axes))
    if np.prod(shape)!=len(a):
        raise ValueError('Expected a complete rectangular grid.')
    ids=tuple(np.searchsorted(x,a[:,i]) for i,x in enumerate(axes))
    flat=np.ravel_multi_index(ids,shape)
    if len(np.unique(flat))!=len(a):
        raise ValueError('Duplicated grid coordinates.')
    f=np.empty(shape);f[ids]=a[:,dimension]*kbt(1,output_unit)/kbt(1,energy_unit)
    f-=np.min(f[np.isfinite(f)])
    unknown=np.full(shape,np.nan)
    return FES(axes,f,unknown.copy(),unknown.copy(),np.zeros(shape,int),
               dict(unit=output_unit,source=str(path),temperature=None,method='imported'))


def average_fes(surfaces,*,reference):
    """Align at the SAME coordinate; return arithmetic mean/std on common support.

    Different MLP Hamiltonians remain different ensembles. This is a descriptive
    model spread (ddof=0), not an exact thermodynamic mixture or standard error.
    """
    first=surfaces[0];values=[]
    for s in surfaces:
        if s.unit!=first.unit or len(s.axes)!=len(first.axes) or any(
            not np.array_equal(a,b) for a,b in zip(s.axes,first.axes)):
            raise ValueError('All surfaces must use the same grid and energy unit.')
        ref=float(s.profile(np.atleast_2d(reference))[0])
        if not np.isfinite(ref):
            raise ValueError('Reference must lie on supported cells in every surface.')
        values.append(s.free_energy-ref)
    values=np.asarray(values);good=np.all(np.isfinite(values),axis=0)
    mean=np.full(first.free_energy.shape,np.nan);std=mean.copy()
    mean[good]=values[:,good].mean(axis=0);std[good]=values[:,good].std(axis=0)
    return mean,std

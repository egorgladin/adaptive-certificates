"""Serial GPU queries, complete CPU masters, and explicit bandwidth emulation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import gc
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .certificates import FullArchive, CompressedArchive, full_qp, compressed_socp
from .oracle import KernelProblem


class Optimizer:
    def __init__(self,x,method,lr):
        self.method,self.lr,self.t = method,lr,0
        self.state = torch.zeros_like(x)

    def step(self,x,g):
        self.t += 1
        if self.method=='subgradient':
            x.add_(g,alpha=-self.lr/math.sqrt(self.t))
        elif self.method=='momentum':
            self.state.mul_(.9).add_(g,alpha=.1)
            x.add_(self.state,alpha=-self.lr/math.sqrt(self.t))
        elif self.method=='adagrad':
            self.state.addcmul_(g,g)
            x.addcdiv_(g,torch.sqrt(self.state)+1e-8,value=-self.lr)
        else:
            raise ValueError(self.method)


class Channel:
    def __init__(self,mib):
        self.bandwidth = mib*2**20 if mib else math.inf
        self.bytes,self.seconds,self.requested = 0,0.,0.

    def send(self,size):
        self.bytes += size
        duration = size/self.bandwidth
        self.requested += duration
        if duration==0:
            return
        start = time.perf_counter()
        deadline = start+duration
        # Avoid Windows sleep quantization on tiny single_cut/compressed messages.
        # Sleep the bulk of long transfers, then finish against a monotonic clock.
        if duration>.002:
            time.sleep(duration-.001)
        while time.perf_counter()<deadline:
            pass
        self.seconds += time.perf_counter()-start


class CappedCompressedArchive(CompressedArchive):
    """Same historical certificate; at capacity retain actual residual radius."""
    def insert(self,a,b):
        before = torch.cuda.memory_allocated()-self.Q.numel()*8
        torch.cuda.reset_peak_memory_stats()
        Q = self.Q[:self.k]
        v = Q@a
        residual = a-v@Q
        norm_tensor = torch.linalg.vector_norm(residual)
        norm = float(norm_tensor)
        if norm>self.tau and self.k<self.capacity:
            q = residual/norm
            for _ in range(2):
                q = q-(Q@q)@Q
            q = q/torch.linalg.vector_norm(q)
            self.Q[self.k].copy_(q)
            self.k += 1
            Q = self.Q[:self.k]
            v = Q@a
            residual = a-v@Q
            norm_tensor = torch.linalg.vector_norm(residual)
        eta = norm_tensor*(1+1e-12)+1e-14
        packet = torch.cat((v,eta[None]))
        self.staging[:self.k+1].copy_(packet,non_blocking=False)
        host = self.staging[:self.k+1].numpy()
        n = len(self.b)
        self.V[n,:self.k] = host[:self.k]
        self.b.append(b)
        self.eta.append(float(host[self.k]))
        self.birth.append(self.k)
        # Certificate service receives k doubles, eta and birth; Q stays on GPU.
        self.wire_bytes += (self.k+2)*8
        self.copies += 1
        self.peak_related = max(self.peak_related,torch.cuda.max_memory_allocated()-before)


def initialize():
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA-enabled PyTorch installation and NVIDIA GPU are required.')
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    torch.use_deterministic_algorithms(True)
    full_qp(np.eye(2),np.ones(2),.1)
    compressed_socp(np.eye(2),np.ones(2),np.zeros(2),np.array([1,2]),.1)


def run(dataset,method,lr,arm,output,records=2000,eps_rel=(.001,),tau=1e-5,
        bandwidth=0.,check_every=32,average_every=16,capacity=256,seconds=600.,
        stop=True,score=True,problem=None):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    problem = problem or KernelProblem(dataset)
    x = torch.zeros(problem.d,dtype=torch.float64,device='cuda')
    optimizer = Optimizer(x,method,lr)
    average,best = torch.zeros_like(x),torch.zeros_like(x)
    problem.oracle(x)
    torch.cuda.synchronize()
    initial = float(problem.oracle(x)[0])
    targets = [initial*eps for eps in eps_rel]
    channel = Channel(bandwidth)
    components = dict(oracle=0.,optimizer=0.,archive=0.,master=0.,monitoring=0.)
    allocated = time.perf_counter()
    archive = FullArchive(problem.d,records,problem.lam) if arm=='full' else CappedCompressedArchive(problem.d,records,problem.lam,tau,min(capacity,problem.d),math.inf) if arm=='compressed' else None
    components['archive'] += time.perf_counter()-allocated
    host = torch.empty(7,dtype=torch.float64,pin_memory=True)
    U,lower,single_cut = math.inf,0.,0.
    n,iterations,count = 0,0,0
    trace,checks,hits = [],[],{}
    intercept = 0.
    status,error = 'record_limit',None
    started = time.perf_counter()-components['archive']

    def record(point):
        nonlocal n,U,lower,single_cut,intercept,status
        t = time.perf_counter()
        F,g,a,packet = problem.oracle(point)
        host.copy_(packet,non_blocking=False)
        scalar = host.numpy().copy()
        components['oracle'] += time.perf_counter()-t
        n += 1
        trace.append(scalar)
        t = time.perf_counter()
        if scalar[0]<U:
            U,intercept = float(scalar[0]),float(scalar[3])
            best.copy_(point)
        single_cut = max(single_cut,float(scalar[0]-scalar[2]/(2*problem.lam)))
        lower = max(lower,single_cut)
        components['monitoring'] += time.perf_counter()-t
        previous_wire = archive.wire_bytes if archive else 0
        if archive:
            t = time.perf_counter()
            archive.insert(a,float(scalar[1]))
            components['archive'] += time.perf_counter()-t
        # Common certificate message: F, support intercept, squared full subgradient.
        channel.send(24+(archive.wire_bytes-previous_wire if archive else 0))
        if archive and (n==1 or n%check_every==0):
            t = time.perf_counter()
            value,info = archive.solve()
            components['master'] += time.perf_counter()-t
            lower = max(lower,value)
            if info['status']!='Solved':
                # Feasible weights still give a valid bound; report solver status.
                info['approximate_master'] = True
            checks.append(dict(records=n,U=U,lower=lower,gap=U-lower,single_cut=single_cut,
                               k=getattr(archive,'k',None),**info))
        elapsed = time.perf_counter()-started
        for eps,target in zip(eps_rel,targets):
            if U-lower<=target and str(eps) not in hits:
                hits[str(eps)] = dict(records=n,seconds=elapsed,nonchannel_seconds=elapsed-channel.seconds,
                                     channel_seconds=channel.seconds,channel_requested_seconds=channel.requested,
                                     bytes=channel.bytes,U=U,lower=lower,gap=U-lower,single_cut_gap=U-single_cut,
                                     k=getattr(archive,'k',None),components=components.copy(),
                                     archive_bytes=archive.numeric_bytes() if archive else 0)
        if stop and U-lower<=min(targets):
            status = 'certified'
        return g

    try:
        while n<records:
            g = record(x)
            if status=='certified':
                break
            t = time.perf_counter()
            average.add_(x)
            count += 1
            optimizer.step(x,g)
            iterations += 1
            torch.cuda.synchronize()
            components['optimizer'] += time.perf_counter()-t
            if average_every and iterations%average_every==0 and n<records:
                g = record(average/count)
                average.zero_(); count=0
                if status=='certified':
                    break
            # Budget applies to actual work; imposed communication is not a cutoff.
            if time.perf_counter()-started-channel.seconds>seconds:
                status = 'time_limit'
                break
    except Exception as exc:
        status,error = 'failure',repr(exc)
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-started
    config = dict(dataset=dataset,method=method,lr=lr,arm=arm,records=records,eps_rel=list(eps_rel),
                  tau=tau,bandwidth_mib=bandwidth,check_every=check_every,average_every=average_every,
                  capacity=capacity,seconds=seconds,momentum=.9,adagrad_offset=1e-8,
                  model_lambda=problem.lam,kernel_sigma=problem.meta['sigma'],nu=.1,stop=stop)
    validation = dict(max_profile_derivative_error=float(np.max(np.abs(np.asarray(trace)[:,-1]))) if trace else None)
    if archive and len(archive.weights):
        archive.weights = np.pad(archive.weights,(0,len(archive.b)-len(archive.weights)))
        validation['simplex_sum_error'] = abs(float(archive.weights.sum())-1)
        validation['min_weight'] = float(archive.weights.min())
        if arm=='full':
            aggregate = archive.direct_aggregate()
            direct = float(np.asarray(archive.b)@archive.weights-aggregate@aggregate/(2*problem.lam))
            validation['gram_direct_error'] = abs(direct-archive.evaluate(archive.weights))
        else:
            if archive.k:
                Q = archive.Q[:archive.k]
                validation['orthogonality_max'] = float(torch.max(torch.abs(Q@Q.T-torch.eye(archive.k,device='cuda'))))
    test_metric = problem.score(best,intercept) if score else None
    result = dict(config=config,status=status,error=error,records=n,iterations=iterations,
                  initial=initial,targets=targets,U=U,lower=lower,gap=U-lower,single_cut=single_cut,
                  first_hits=hits,checks=checks,components=components,seconds=elapsed,
                  nonchannel_seconds=elapsed-channel.seconds,channel_seconds=channel.seconds,
                  channel_requested_seconds=channel.requested,bytes=channel.bytes,
                  archive_bytes=archive.numeric_bytes() if archive else 0,k=getattr(archive,'k',None),
                  gpu_archive_peak_bytes=archive.peak_related if arm=='compressed' else 0,
                  test_metric=test_metric,intercept=intercept,validation=validation,
                  environment=dict(torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),
                                   dtype='float64',cpu_threads=1,channel='synchronous additional remote link; L=0; bytes/B',
                                   timers='solve only; imports/preparation/scoring/file writes excluded'),
                  )
    np.save(output/'trace.npy',np.asarray(trace))
    (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(cell=str(output),status=status,FG=n,gap=U-lower,k=getattr(archive,'k',None),
                          seconds=elapsed,channel=channel.seconds,error=error)),flush=True)
    del archive,optimizer,x,average,best
    gc.collect()
    return result


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset',required=True)
    p.add_argument('--method',required=True,choices=('subgradient','momentum','adagrad'))
    p.add_argument('--lr',type=float,required=True)
    p.add_argument('--arm',required=True,choices=('single_cut','full','compressed'))
    p.add_argument('--output',required=True)
    p.add_argument('--records',type=int,default=2000)
    p.add_argument('--eps-rel',type=float,nargs='+',default=[.001])
    p.add_argument('--tau',type=float,default=1e-5)
    p.add_argument('--bandwidth',type=float,default=0.)
    p.add_argument('--check-every',type=int,default=32)
    p.add_argument('--capacity',type=int,default=256)
    p.add_argument('--seconds',type=float,default=600)
    args = p.parse_args()
    initialize()
    run(**vars(args))

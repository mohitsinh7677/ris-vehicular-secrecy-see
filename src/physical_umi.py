"""Channel, power, benchmark and TD3 models for the RIS-assisted vehicular secrecy study.
Simplified model: 3GPP TR 38.901 UMi path loss for direct links, geometric Rician RIS hops;
not a full spatially consistent 3GPP TR 38.901 channel emulator.

Direct BS-UE links use UMi Street Canyon LOS/NLOS large-scale formulae (3GPP TR 38.901).
RIS hops use simplified geometric LOS Friis/array-fading assumptions; no 3GPP cluster engine.
The TD3 helper in this file trains with terminal one-step transitions (contextual bandit).
"""
from __future__ import annotations
import argparse, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from scipy.special import j0
from scipy.linalg import eigh
import torch
from torch import nn
import torch.nn.functional as F

@dataclass(frozen=True)
class Config:
    seed:int=20261010
    n:int=128
    m:int=8
    fc_ghz:float=28.0
    bw_hz:float=100e6
    noise_figure_db:float=7.0
    pt_dbm:float=15.0
    v_kmh:float=60.0
    csi_std:float=0.05
    delay_ms:float=0.10
    bs:tuple=(0.0,0.0,10.0)
    ris:tuple=(80.0,10.0,6.0)
    user:tuple=(100.0,0.0,1.5)
    eve:tuple=(100.0,-10.0,1.5)
    bs_gain_dbi:float=10.0   # fixed assumed per-antenna pattern gain
    ris_hop_gain_dbi:float=6.0   # assumed directional element gain on EACH hop
    k_ris_db:float=5.0
    pa_eff:float=0.35  # ASSUMED / needs literature-hardware calibration
    bs_circuit_w:float=0.50
    ris_ctrl_w:float=0.10
    per_element_w:float=0.001
    ao_iterations:int=12
    def pt_w(self): return 10.0**((self.pt_dbm-30.)/10.)
    def noise_dbm(self): return -174.0+10.0*np.log10(self.bw_hz)+self.noise_figure_db
    def noise_w(self): return 10.0**((self.noise_dbm()-30)/10.)
    def total_power_w(self,ris=True):
        return self.pt_w()/self.pa_eff+self.bs_circuit_w+(self.ris_ctrl_w+self.n*self.per_element_w if ris else 0.0)
    def max_doppler(self):return (self.v_kmh/3.6)*(self.fc_ghz*1e9)/299792458.0
    def rho(self):return float(j0(2*np.pi*self.max_doppler()*self.delay_ms/1000.0))

def db2pow(x):return 10.**(-x/10.)
def normpath(d):return float(max(d,10.))
def pl_umi_los(d,fc): return 32.4+21.*np.log10(normpath(d))+20.*np.log10(fc)
def pl_umi_nlos(d,fc,h_ue=1.5):
    return max(pl_umi_los(d,fc), 22.4+35.3*np.log10(normpath(d))+21.3*np.log10(fc)-.3*(h_ue-1.5))
def pl_hop_los(d,fc):return 32.4+20.*np.log10(normpath(d))+20.*np.log10(fc)
def dist(x,y):return float(np.linalg.norm(np.array(x)-np.array(y)))
def complex_rand(rng,shape):return (rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)
def ula(k,az):return np.exp(1j*np.pi*np.arange(k)*np.sin(az))

def ris_upa(n, source, target):
    """Half-wavelength rectangular RIS in the yz-plane, with n elements.
    A balanced integer factorization is used for N=32,64,128,256.
    Plane-wave steering is applicable under the nominal far-field geometry.
    This is not a 3GPP clustered/spatially-consistent RIS channel generator.
    """
    rows=int(np.sqrt(n))
    while n%rows: rows-=1
    cols=n//rows
    direction=(np.asarray(target,dtype=float)-np.asarray(source,dtype=float))
    direction=direction/np.linalg.norm(direction)
    zz,yy=np.meshgrid(np.arange(rows),np.arange(cols),indexing='ij')
    return np.exp(1j*np.pi*(yy*direction[1]+zz*direction[2])).reshape(n)

def far_field_distance_m(cfg):
    wl=299792458./(cfg.fc_ghz*1e9)
    rows=int(np.sqrt(cfg.n))
    while cfg.n%rows: rows-=1
    cols=cfg.n//rows
    # Fraunhofer criterion uses the largest physical aperture dimension.
    # For the rectangular panel, this is the diagonal between outer element centers.
    D=np.hypot(rows-1,cols-1)*wl/2.
    return 2*D**2/wl


def rician(rng,shape,k_db,los):
    K=10.**(k_db/10.)
    return np.sqrt(K/(K+1))*los+np.sqrt(1/(K+1))*complex_rand(rng,shape)

def channel_snapshot(cfg:Config,rng):
    """Generate complete true + noisy delayed legitimate CSI; Eve CSI never observed.
    Each sample is an INDEPENDENT snapshot. Only moving-user links age with Doppler.
    """
    m,n=cfg.m,cfg.n
    d_bu=dist(cfg.bs,cfg.user);d_be=dist(cfg.bs,cfg.eve)
    d_br=dist(cfg.bs,cfg.ris);d_ru=dist(cfg.ris,cfg.user);d_re=dist(cfg.ris,cfg.eve)
    Gbs=10**(cfg.bs_gain_dbi/10.)
    Gris=10**(cfg.ris_hop_gain_dbi/10.)
    # Blocked direct U (NLOS) and passive Eve NLOS, with spatially isotropic Eve scatter.
    beta_bu=db2pow(pl_umi_nlos(d_bu,cfg.fc_ghz))*Gbs
    beta_be=db2pow(pl_umi_nlos(d_be,cfg.fc_ghz))*Gbs
    # Simplified RIS hops: LOS free-space/geometric attenuation + directional element gain.
    beta_br=db2pow(pl_hop_los(d_br,cfg.fc_ghz))*Gbs*Gris
    beta_ru=db2pow(pl_hop_los(d_ru,cfg.fc_ghz))*Gris
    beta_re=db2pow(pl_umi_nlos(d_re,cfg.fc_ghz))*Gris
    ang_bs=np.arctan2(cfg.ris[1]-cfg.bs[1],cfg.ris[0]-cfg.bs[0])
    ang_ris=np.arctan2(cfg.user[1]-cfg.ris[1],cfg.user[0]-cfg.ris[0])
    br_los=np.outer(ris_upa(n,cfg.ris,cfg.bs),np.conj(ula(m,ang_bs)))
    ru_los=ris_upa(n,cfg.ris,cfg.user)
    Hbr=np.sqrt(beta_br)*rician(rng,(n,m),cfg.k_ris_db,br_los)
    # Old and new physical channels evolve; rho may be negative under the Jakes model.
    rho=cfg.rho()
    hbu_old=np.sqrt(beta_bu)*complex_rand(rng,(m,))
    # Rician process: Doppler-phase-rotating deterministic LOS + Jakes-aged diffuse part.
    kr=10.**(cfg.k_ris_db/10.)
    scat_old=complex_rand(rng,(n,))
    hru_old=np.sqrt(beta_ru)*(np.sqrt(kr/(kr+1))*ru_los+np.sqrt(1/(kr+1))*scat_old)
    hbu_true=rho*hbu_old+np.sqrt(max(0.,1-rho*rho))*np.sqrt(beta_bu)*complex_rand(rng,(m,))
    path_cos=(cfg.user[0]-cfg.ris[0])/dist(cfg.ris,cfg.user)
    los_rot=np.exp(1j*2*np.pi*cfg.max_doppler()*cfg.delay_ms/1000.*path_cos)
    scat_true=rho*scat_old+np.sqrt(max(0.,1-rho*rho))*complex_rand(rng,(n,))
    hru_true=np.sqrt(beta_ru)*(np.sqrt(kr/(kr+1))*ru_los*los_rot+np.sqrt(1/(kr+1))*scat_true)
    hbe_true=np.sqrt(beta_be)*complex_rand(rng,(m,))
    hre_true=np.sqrt(beta_re)*complex_rand(rng,(n,))
    hbu_obs=hbu_old+np.sqrt(beta_bu)*cfg.csi_std*complex_rand(rng,(m,))
    hru_obs=hru_old+np.sqrt(beta_ru)*cfg.csi_std*complex_rand(rng,(n,))
    Hbr_obs=Hbr+np.sqrt(beta_br)*cfg.csi_std*complex_rand(rng,(n,m))
    return {'beta_be':beta_be,'beta_re':beta_re,'hbu_true':hbu_true,'hru_true':hru_true,'hbe_true':hbe_true,'hre_true':hre_true,
            'Hbr_true':Hbr,'hbu_obs':hbu_obs,'hru_obs':hru_obs,'Hbr_obs':Hbr_obs,
            'scales':np.array([np.sqrt(beta_bu),np.sqrt(beta_ru),np.sqrt(beta_br)])}

def effective_row(hb,hr,H,phi):
    return hb.conj()+np.einsum('n,n,nm->m',hr.conj(),phi,H)

def get_mrt(row,pt):
    # row = g^H. MRT w = conjugate(g^H) * sqrt(P) / norm(g)
    z=np.conj(row)
    norm=np.linalg.norm(z)
    if norm<1e-30:z=np.ones(len(z),dtype=complex);norm=np.linalg.norm(z)
    return np.sqrt(pt)*z/norm

def secrecy(cfg,sample,w,phi=None):
    rowu=sample['hbu_true'].conj() if phi is None else effective_row(sample['hbu_true'],sample['hru_true'],sample['Hbr_true'],phi)
    rowe=sample['hbe_true'].conj() if phi is None else effective_row(sample['hbe_true'],sample['hre_true'],sample['Hbr_true'],phi)
    gu=float(abs(rowu@w)**2/cfg.noise_w());ge=float(abs(rowe@w)**2/cfg.noise_w())
    return max(0.,float(np.log2(1+gu)-np.log2(1+ge))),gu,ge

def eve_covariance(cfg,sample,ris=True):
    # For isotropic zero-mean passive-Eve channels, expected Eve |gE^H w|^2
    # is w^H C_E w, where RIS element phases cancel in expectation.
    C=sample['beta_be']*np.eye(cfg.m,dtype=complex)
    if ris:
        H=sample['Hbr_obs']
        C=C+sample['beta_re']*(H.conj().T@H)
    return C

def secrecy_surrogate_beamformer(cfg,legit_row,C):
    # Maximize ratio (1+estimated-user SNR)/(1+expected-Eve SNR), fixed Pt.
    g=np.conj(legit_row)
    I=np.eye(cfg.m,dtype=complex)
    A=I+(cfg.pt_w()/cfg.noise_w())*np.outer(g,np.conj(g))
    B=I+(cfg.pt_w()/cfg.noise_w())*C
    _,evec=eigh(A,B,check_finite=False)
    z=evec[:,-1];return np.sqrt(cfg.pt_w())*z/np.linalg.norm(z)

def ao_action(cfg,sample):
    # Equal-information passive-Eve statistical secrecy surrogate.
    # AO phase step maximizes observed-U amplitude for fixed beam; Eve expected
    # leakage is phase-invariant under the assumed isotropic h_RE distribution.
    hb,hr,H=sample['hbu_obs'],sample['hru_obs'],sample['Hbr_obs']
    C=eve_covariance(cfg,sample,True)
    phi=np.ones(cfg.n,dtype=complex)
    for _ in range(cfg.ao_iterations):
        w=secrecy_surrogate_beamformer(cfg,effective_row(hb,hr,H,phi),C)
        q=np.conj(hr)*(H@w)
        direct=np.vdot(hb,w)
        phase0=np.angle(direct) if abs(direct)>1e-22 else 0.
        phi=np.exp(1j*(phase0-np.angle(q)))
    w=secrecy_surrogate_beamformer(cfg,effective_row(hb,hr,H,phi),C)
    return w,phi

def random_action(cfg,sample,rng):
    phi=np.exp(1j*rng.uniform(-np.pi,np.pi,cfg.n))
    C=eve_covariance(cfg,sample,True)
    return secrecy_surrogate_beamformer(cfg,effective_row(sample['hbu_obs'],sample['hru_obs'],sample['Hbr_obs'],phi),C),phi

def observation(cfg,s):
    hb=s['hbu_obs']/s['scales'][0]
    hr=s['hru_obs']/s['scales'][1]
    H=s['Hbr_obs']/s['scales'][2]
    # Full legitimate imperfect/delayed channel observation. Eve excluded.
    z=np.concatenate([hb.ravel(),hr.ravel(),H.ravel()])
    x=np.concatenate([z.real,z.imag,np.array([cfg.pt_dbm/30,cfg.v_kmh/120,cfg.csi_std/.20,cfg.delay_ms/1.])])
    return x.astype(np.float32)

def physical_action(cfg,a):
    a=np.asarray(a,dtype=float)
    assert len(a)==2*cfg.m+cfg.n
    b=a[:cfg.m]+1j*a[cfg.m:2*cfg.m]
    bn=np.linalg.norm(b)
    if bn<1e-12:b=np.ones(cfg.m,dtype=complex);bn=np.linalg.norm(b)
    w=np.sqrt(cfg.pt_w())*b/bn
    phi=np.exp(1j*np.pi*a[2*cfg.m:])
    return w,phi

class MLP(nn.Module):
    def __init__(self,dim_in,dim_out,out_tanh=False):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim_in,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),nn.Linear(256,dim_out))
        self.out_tanh=out_tanh
    def forward(self,x):
        z=self.net(x)
        return torch.tanh(z) if self.out_tanh else z

class TD3:
    def __init__(self,sd,ad,seed=10):
        torch.manual_seed(seed)
        self.actor=MLP(sd,ad,True);self.actor_target=MLP(sd,ad,True)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.q1=MLP(sd+ad,1);self.q2=MLP(sd+ad,1)
        self.q1t=MLP(sd+ad,1);self.q2t=MLP(sd+ad,1)
        self.q1t.load_state_dict(self.q1.state_dict());self.q2t.load_state_dict(self.q2.state_dict())
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=3e-4)
        self.q_opt=torch.optim.Adam(list(self.q1.parameters())+list(self.q2.parameters()),lr=3e-4)
        self.step=0
    @torch.no_grad()
    def act(self,obs):return self.actor(torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)).numpy()[0]
    def update(self,batch,rand):
        s,a,r,s2,done=batch
        st=torch.as_tensor(s,dtype=torch.float32);at=torch.as_tensor(a,dtype=torch.float32)
        rt=torch.as_tensor(r,dtype=torch.float32).view(-1,1)
        st2=torch.as_tensor(s2,dtype=torch.float32);dt=torch.as_tensor(done,dtype=torch.float32).view(-1,1)
        with torch.no_grad():
            n=torch.clamp(torch.randn_like(at)*.2,-.5,.5)
            a2=torch.clamp(self.actor_target(st2)+n,-1.,1.)
            qtar=rt+.99*(1.-dt)*torch.minimum(self.q1t(torch.cat([st2,a2],-1)),self.q2t(torch.cat([st2,a2],-1)))
        xin=torch.cat([st,at],dim=1)
        loss=F.mse_loss(self.q1(xin),qtar)+F.mse_loss(self.q2(xin),qtar)
        self.q_opt.zero_grad();loss.backward();self.q_opt.step()
        self.step+=1
        aloss=None
        if self.step%2==0:
            aloss=-self.q1(torch.cat([st,self.actor(st)],-1)).mean()
            self.actor_opt.zero_grad();aloss.backward();self.actor_opt.step()
            with torch.no_grad():
                for network,target in ((self.actor,self.actor_target),(self.q1,self.q1t),(self.q2,self.q2t)):
                    for p,pt in zip(network.parameters(),target.parameters()):pt.data.mul_(.995).add_(.005*p.data)
        return float(loss.item()),(float(aloss.item()) if aloss is not None else None)

class Replay:
    def __init__(self,capacity,sd,ad):
        self.capacity=capacity;self.i=0;self.count=0
        self.s=np.empty((capacity,sd),np.float32);self.a=np.empty((capacity,ad),np.float32)
        self.r=np.empty(capacity,np.float32);self.s2=np.empty((capacity,sd),np.float32);self.d=np.empty(capacity,np.float32)
    def add(self,s,a,r,s2,d):
        i=self.i;self.s[i]=s;self.a[i]=a;self.r[i]=r;self.s2[i]=s2;self.d[i]=d
        self.i=(i+1)%self.capacity;self.count=min(self.count+1,self.capacity)
    def batch(self,rng,b):
        i=rng.integers(0,self.count,b)
        return (self.s[i],self.a[i],self.r[i],self.s2[i],self.d[i])

def evaluate(cfg,agent,seed=8820,count=1000):
    rng=np.random.default_rng(seed)
    records=[]
    times={'AO-RIS':0.0,'TD3-RIS':0.0,'Random-RIS':0.0,'No-RIS':0.0}
    for i in range(count):
        s=channel_snapshot(cfg,rng)
        rr={}
        t0=time.perf_counter();w,p=ao_action(cfg,s);times['AO-RIS']+=time.perf_counter()-t0
        rr['AO-RIS']=secrecy(cfg,s,w,p)[0]
        t0=time.perf_counter();w,p=random_action(cfg,s,rng);times['Random-RIS']+=time.perf_counter()-t0
        rr['Random-RIS']=secrecy(cfg,s,w,p)[0]
        t0=time.perf_counter();w=secrecy_surrogate_beamformer(cfg,s['hbu_obs'].conj(),eve_covariance(cfg,s,False));times['No-RIS']+=time.perf_counter()-t0
        rr['No-RIS']=secrecy(cfg,s,w,None)[0]
        t0=time.perf_counter();a=agent.act(observation(cfg,s));w,p=physical_action(cfg,a);times['TD3-RIS']+=time.perf_counter()-t0
        rr['TD3-RIS']=secrecy(cfg,s,w,p)[0]
        records.append(rr)
    return records,{m:1e3*t/count for m,t in times.items()}

def train_one_step(cfg,steps,grad_steps,seed):
    """Independent channels with terminal transitions: one-step TD3 baseline (contextual bandit), not a sequential vehicular MDP."""
    rng=np.random.default_rng(seed)
    torch.set_num_threads(1)
    sd=len(observation(cfg,channel_snapshot(cfg,rng)));ad=2*cfg.m+cfg.n
    agent=TD3(sd,ad,seed);buffer=Replay(max(1000,steps+5),sd,ad)
    returns=[];losses=[]
    for t in range(steps):
        sn=channel_snapshot(cfg,rng);ob=observation(cfg,sn)
        if t<max(80,steps//5):a=rng.uniform(-1,1,ad)
        else:a=np.clip(agent.act(ob)+rng.normal(0,.1,ad),-1.,1.)
        w,p=physical_action(cfg,a)
        rew=secrecy(cfg,sn,w,p)[0]
        # terminal transition: the agent learns a one-step channel-to-action mapping
        buffer.add(ob,a,rew,ob,1.)
        returns.append(rew)
        if t>=max(80,steps//5) and t<steps and (t % max(1,steps//max(1,grad_steps))==0):
            loss,_=agent.update(buffer.batch(rng,min(128,buffer.count)),rng)
            losses.append(loss)
    return agent, {'transitions':steps,'updates':len(losses),'mean_recent_train_reward':float(np.mean(returns[-min(100,steps):])),
                   'last_critic_loss':float(losses[-1]) if losses else None,'finite_losses':bool(np.isfinite(losses).all())}

def run(args):
    cfg=Config()
    out=Path(args.output);out.mkdir(exist_ok=True,parents=True)
    rng=np.random.default_rng(cfg.seed)
    sam=channel_snapshot(cfg,rng)
    unit_action=np.zeros(2*cfg.m+cfg.n)
    w,phi=physical_action(cfg,unit_action)
    assert np.isclose(np.vdot(w,w).real,cfg.pt_w(),rtol=1e-12)
    assert np.max(np.abs(np.abs(phi)-1))<1e-12
    assert np.isfinite(secrecy(cfg,sam,w,phi)[0])
    assert cfg.total_power_w(True)>cfg.total_power_w(False)>cfg.pt_w()
    assert np.isclose(cfg.noise_dbm(),-87,atol=1e-6)
    assert abs(cfg.max_doppler()-1556.6)<5.0
    assert np.isfinite(cfg.rho())
    t0=time.perf_counter()
    agent,train=train_one_step(cfg,args.steps,args.grad_steps,cfg.seed+1)
    training_sec=time.perf_counter()-t0
    t0=time.perf_counter();records,timings=evaluate(cfg,agent,count=args.test_channels);eval_sec=time.perf_counter()-t0
    import csv
    with (out/'matched_channel_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
    methods=list(records[0]);stats={}
    for met in methods:
        vals=np.array([r[met] for r in records]);pwr=cfg.total_power_w(met!='No-RIS')
        stats[met]={'n':len(vals),'asr_mean_bpshz':float(vals.mean()),'asr_sd_bpshz':float(vals.std(ddof=1)),
                    'asr_95ci_halfwidth':float(1.96*vals.std(ddof=1)/np.sqrt(len(vals))),
                    'see_bit_per_j_hz':float(vals.mean()/pwr),'system_power_w_assumed':pwr,'compute_time_ms_mean':float(timings[met])}
    result={'status':'Nominal single-point run (legacy helper); simplified channel model','config':asdict(cfg),
            'derived':{'pt_w':cfg.pt_w(),'noise_dbm':cfg.noise_dbm(),'noise_w':cfg.noise_w(),
                       'max_doppler_hz':cfg.max_doppler(),'coherence_estimate_ms':1000*.423/cfg.max_doppler(),
                       'jakes_rho_at_delay':cfg.rho(),'power_w_ris_assumed':cfg.total_power_w(),
                       'power_w_no_ris_assumed':cfg.total_power_w(False),
                       'pl_bs_user_nlos_db':pl_umi_nlos(dist(cfg.bs,cfg.user),cfg.fc_ghz),
                       'pl_bs_ris_los_db':pl_hop_los(dist(cfg.bs,cfg.ris),cfg.fc_ghz),
                       'pl_ris_user_los_db':pl_hop_los(dist(cfg.ris,cfg.user),cfg.fc_ghz)},
            'training_one_step':train,'timing':{'train_seconds':training_sec,'evaluation_seconds':eval_sec},
            'benchmarks':stats,
            'limitations':['3GPP UMi pathloss expressions only, not validated 3GPP TR 38.901 cluster/spatial consistency',
                            'Rician LOS evolves through Doppler phase and diffuse component uses Jakes aging; still not a 3GPP spatially-consistent ray model',
                            'Power coefficients and geometry are working assumptions not calibrated to hardware',
                            'TD3 uses independent terminal snapshots (contextual bandit), not a sequential MDP',
                            'Short TD3 training cannot evidence R7 convergence or performance superiority',
                            'AO optimizes expected-Eve-SNR secrecy surrogate under isotropic passive Eve and observed legitimate CSI, not exact expected secrecy rate; robust/multi-start AO needed before publication']}
    with (out/'nominal_summary.json').open('w') as f:json.dump(result,f,indent=2)
    torch.save({'actor':agent.actor.state_dict(),'config':asdict(cfg),'observation_dim':len(observation(cfg,sam))},out/'td3_one_step_actor.pt')
    print(json.dumps({'derived':result['derived'],'training':train,'benchmarks':stats,'runtime':result['timing']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs');p.add_argument('--steps',type=int,default=500)
    p.add_argument('--grad-steps',type=int,default=200);p.add_argument('--test-channels',type=int,default=1000)
    run(p.parse_args())

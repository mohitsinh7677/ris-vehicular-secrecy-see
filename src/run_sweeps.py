"""Sensitivity sweeps for the RIS-assisted vehicular secrecy study.

Trains five TD3 baseline policies per RIS size with terminal one-step transitions
(contextual-bandit formulation) and evaluates TD3, AO-RIS, Random-RIS and No-RIS on
1,000 matched channel snapshots per operating point. Produces the CSV records behind
Figs. 1-7 and Table 2 of the paper. Figures are drawn by revision/make_figures.py.
"""
import sys, os, json, csv, time, argparse
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
import torch

from physical_umi import Config, channel_snapshot, observation, TD3, Replay, physical_action, secrecy, evaluate, ao_action, random_action, secrecy_surrogate_beamformer, eve_covariance, pl_umi_los, pl_umi_nlos, dist

N_VALUES=[32,64,128,256]
P_VALUES=[0,5,10,15,20,25,30]
SIGMAS=[0,.02,.05,.1,.15,.2]
SPEEDS=[30,45,60,75,90,105,120]
DELAYS=[0,.1,.25,.5,.75,1.]
METHODS=['TD3-RIS','AO-RIS','Random-RIS','No-RIS']


def sample_training_config(cfg,rng):
    """Training v <= 90 km/h. User positions random independent snapshots, NOT trajectories."""
    return replace(cfg,
        pt_dbm=float(rng.choice(P_VALUES)),
        v_kmh=float(rng.choice([30.,45.,60.,75.,90.])),
        delay_ms=float(rng.choice(DELAYS)),
        csi_std=float(rng.choice(SIGMAS)),
        user=(float(rng.uniform(60,120)),0.,1.5))


def quick_eval(cfg,agent,inputs):
    """Matched independent samples; uses true Eve channel only for scoring."""
    v=[]
    with torch.no_grad():
        # Batched NN forward removes Python overhead in validation
        arr=np.stack([observation(cfg,s) for s in inputs])
        actions=agent.actor(torch.tensor(arr)).numpy()
    for s,a in zip(inputs,actions):
        w,p=physical_action(cfg,a)
        v.append(secrecy(cfg,s,w,p)[0])
    return float(np.mean(v))


def train_short(cfg,seed,steps=1200,updates=240,checkpoint=200,val_count=120):
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    sam=channel_snapshot(cfg,rng)
    sd=len(observation(cfg,sam));ad=2*cfg.m+cfg.n
    agent=TD3(sd,ad,seed)
    mem=Replay(max(steps+1,2000),sd,ad)
    validation_rng=np.random.default_rng(440011+cfg.n)
    validation=[channel_snapshot(cfg,validation_rng) for _ in range(val_count)]
    checkpoints=[(0,quick_eval(cfg,agent,validation))]
    losses=[]
    warm=max(150,steps//5)
    for t in range(steps):
        ct=sample_training_config(cfg,rng)
        s=channel_snapshot(ct,rng)
        ob=observation(ct,s)
        a=rng.uniform(-1.,1.,ad) if t <warm else np.clip(agent.act(ob)+rng.normal(0,.1,ad),-1,1)
        w,p=physical_action(ct,a)
        reward=secrecy(ct,s,w,p)[0]
        mem.add(ob,a,reward,ob,1.)  # terminal: single-step contextual bandit, NOT a trajectory
        if t>=warm and (t-warm)%max(1,(steps-warm)//updates)==0:
            loss,_=agent.update(mem.batch(rng,min(128,mem.count)),rng)
            losses.append(loss)
        if (t+1)%checkpoint==0:
            checkpoints.append((t+1,quick_eval(cfg,agent,validation)))
    assert np.all(np.isfinite(np.array(losses)))
    return agent,checkpoints,len(losses)


def matched(cfg,agents,channel_seed=12345,n_mc=1000):
    rng=np.random.default_rng(channel_seed)
    vals={m:np.empty(n_mc) for m in METHODS}
    times={m:0. for m in METHODS}
    for i in range(n_mc):
        s=channel_snapshot(cfg,rng)
        t=time.perf_counter();w,p=ao_action(cfg,s);times['AO-RIS']+=time.perf_counter()-t
        vals['AO-RIS'][i]=secrecy(cfg,s,w,p)[0]
        t=time.perf_counter();w,p=random_action(cfg,s,rng);times['Random-RIS']+=time.perf_counter()-t
        vals['Random-RIS'][i]=secrecy(cfg,s,w,p)[0]
        t=time.perf_counter();w=secrecy_surrogate_beamformer(cfg,s['hbu_obs'].conj(),eve_covariance(cfg,s,False));times['No-RIS']+=time.perf_counter()-t
        vals['No-RIS'][i]=secrecy(cfg,s,w)[0]
        t=time.perf_counter()
        ob=torch.from_numpy(observation(cfg,s))[None,:]
        with torch.no_grad():actions=torch.stack([agent.actor(ob)[0] for agent in agents]).numpy()
        for k,a in enumerate(actions):
            w,p=physical_action(cfg,a)
            if k==0: val=[]
            val.append(secrecy(cfg,s,w,p)[0])
        vals['TD3-RIS'][i]=float(np.mean(val))
        times['TD3-RIS']+=time.perf_counter()-t
    # TD3 here is mean over several agents for each realization, not a single model
    return vals,{k:v/n_mc for k,v in times.items()}


def prepare_sweep(sweep,values,cfg,agents,mc):
    rows=[];records=[]
    for x in values:
        ccfg=replace(cfg, **{sweep:float(x) if sweep!='n' else int(x)})
        aa=agents[int(x)] if sweep=='n' else agents[128]
        res,timings=matched(ccfg,aa,channel_seed=880021+int(x*1000)+len(sweep)*90,n_mc=mc)
        for m,v in res.items():
            pw=ccfg.total_power_w(m!='No-RIS')
            row={'sweep':sweep,'x':x,'method':m,'n_channels':mc,
                 'mean_secrecy':float(v.mean()),'sd_channel':float(v.std(ddof=1)),
                 'ci95_halfwidth_channel':float(1.96*v.std(ddof=1)/np.sqrt(mc)),
                 'mean_see':float(v.mean()/pw),'power_w':pw,
                 'cpu_ms_per_decision':1000*timings[m]}
            rows.append(row)
        # Paired AO difference on same channels. TD3 mean across seeds at each channel.
        delta=res['TD3-RIS']-res['AO-RIS']
        records.append({'sweep':sweep,'x':x,'td3_minus_ao_mean':float(delta.mean()),
                        'paired_ci95_halfwidth':float(1.96*delta.std(ddof=1)/np.sqrt(mc))})
    return rows,records


def save_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)



def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--updates',type=int,default=240);ap.add_argument('--mc',type=int,default=1000)
    ap.add_argument('--seeds',type=int,default=5);ap.add_argument('--out',default='results/main')
    args=ap.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    cfg=Config()
    agents={};conv=[];logs=[];start=time.perf_counter()
    for n in N_VALUES:
        agents[n]=[]
        for i in range(args.seeds):
            ag,curve,nupd=train_short(replace(cfg,n=n),seed=19542+101*i+13*n,steps=args.steps,updates=args.updates)
            agents[n].append(ag)
            if n==128:conv.append(curve)
            logs.append({'ris_n':n,'seed_idx':i,'actual_one_step_transitions':args.steps,
                         'critic_updates':nupd,'heldout_reward_last':curve[-1][1]})
        print('trained N',n,'elapsed',round(time.perf_counter()-start,1),'sec',flush=True)
    assert all(len(ag)==args.seeds for ag in agents.values())
    assert len(conv)==args.seeds
    # For comparisons use all five seeds; power=20 for R1, power=15 for R3
    sweeps=[('n',N_VALUES,20.),('pt_dbm',P_VALUES,15.),('n',N_VALUES,15.),
            ('csi_std',SIGMAS,15.),('v_kmh',SPEEDS,15.),('delay_ms',DELAYS,15.)]
    rows=[];differences=[]
    for sweep,values,pt in sweeps:
        c=replace(cfg,pt_dbm=pt)
        r,d=prepare_sweep(sweep,values,c,agents,args.mc)
        for row in r:row['pt_dbm_nominal']=pt
        for row in d:row['pt_dbm_nominal']=pt
        rows+=r;differences+=d
        print('sweep',sweep,'at power',pt,'elapsed',round(time.perf_counter()-start,1),flush=True)
    save_csv(out/'measured_sweep_summary.csv',rows)
    save_csv(out/'paired_td3_ao_differences.csv',differences)
    save_csv(out/'training_audit.csv',logs)
    convrows=[{'seed_idx':i,'transitions':x,'heldout_secrecy':val} for i,hist in enumerate(conv) for x,val in hist]
    save_csv(out/'five_seed_training_checkpoints.csv',convrows)
    summary={'status':'Sensitivity sweeps: five TD3 baseline policies per RIS size trained with terminal one-step transitions; simplified channel model (see README).',
             'config':asdict(cfg),'runs':len(logs),'short_transitions_per_seed_per_N':args.steps,'training_seed_count':args.seeds,
             'sample_channels_per_point':args.mc,'runtime_sec':time.perf_counter()-start,
             'sweep_count':len(sweeps),'test_speeds_not_in_training':[105,120],
             'notes':['Channel is simplified fixed-geometry Rician / 3GPP UMi direct-link pathloss; no longitudinal trajectory or spatial consistency',
                      'Five independent TD3 baseline policies trained per N using terminal one-step rewards (contextual-bandit formulation)',
                      'Eve covariance matched only statistically; true Eve used for offline score',
                      'PA/BS/RIS power constants assumed, not hardware calibrated',
                      'Intervals in this file are channel-sampling 95% normal intervals; seed-inclusive intervals are computed by revision/stats.py']}
    (out/'audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'elapsed_sec':summary['runtime_sec'],'rows':len(rows),'audit':summary['status']},indent=2),flush=True)

if __name__=='__main__': main()

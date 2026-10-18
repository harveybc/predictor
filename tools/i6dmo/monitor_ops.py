"""Install finite monitor campaign timers from an operator-owned private config."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile

from tools.i6dmo import weekly18 as W, monitor_campaign as C


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, **kwargs)


def environment(host, unit, runtime):
    raw = subprocess.check_output(['ssh', host, 'systemctl', '--user', 'show', unit,
                                   '--property=Environment', '--value'], text=True)
    allowed = ('LD_LIBRARY_PATH','CUDA_VISIBLE_DEVICES','XLA_FLAGS','TF_NUM_INTEROP_THREADS',
               'TF_NUM_INTRAOP_THREADS','PYTHONNOUSERSITE','PYTHONSAFEPATH')
    env = dict(item.split('=',1) for item in shlex.split(raw) if item.split('=',1)[0] in allowed)
    if not env.get('CUDA_VISIBLE_DEVICES') or not env.get('LD_LIBRARY_PATH'):
        raise ValueError('WORKER_ENVIRONMENT_MISSING')
    env.update(PYTHONPATH=runtime, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    return env


def service(command, runtime, env, *, memory, secret=None, timeout=3700):
    lines = ['[Unit]', 'Description=Weekly monitor contrast', '[Service]', 'Type=oneshot',
             'WorkingDirectory='+runtime, 'MemoryMax='+memory, 'MemorySwapMax=0',
             'TimeoutStartSec='+str(timeout), 'Nice=10']
    if secret:
        lines.append('EnvironmentFile='+secret)
    lines += ['Environment='+json.dumps(k+'='+v) for k,v in env.items()]
    # systemd uses its own quoting; no shell expansion is involved.
    lines.append('ExecStart='+' '.join(json.dumps(v.replace('%','%%')) for v in command))
    return '\n'.join(lines)+'\n'


def timer():
    return '[Unit]\nDescription=Finite weekly monitor work\n[Timer]\nOnBootSec=30\nOnUnitInactiveSec=30\nAccuracySec=1\n[Install]\nWantedBy=timers.target\n'


def deploy(config):
    root = Path(config['root']); root.mkdir(parents=True, exist_ok=True)
    doc = C.checked_doc(Path(config['packet'])/'PREPARED.json')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=W.REPO,text=True).strip():
        raise ValueError('COMMITTED_SOURCE_REQUIRED')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=W.REPO,text=True).strip()
    names = subprocess.check_output(['git','ls-files','*.py'],cwd=W.REPO,text=True).splitlines()
    code = {name:W.file_sha256(W.REPO/name) for name in names}
    spec = dict(config,code=code,code_commit=commit,outbox=str(root/'outbox.sqlite3'))
    C.seal(root/'DEPLOY_IDENTITY.json',dict(packet=doc['sha256'],code=code,config=config))
    W.B.put(root/'RUN.private.json',spec)
    with tempfile.TemporaryDirectory(prefix='monitor-deploy-') as tmp:
        tmp = Path(tmp); archive = tmp/'code.tar'
        run(['git','archive','--format=tar','--output='+str(archive),'HEAD'],cwd=W.REPO)
        for family,role in spec['roles'].items():
            host = role['host']; runtime = role['runtime']
            env = environment(host,role['source_unit'],runtime)
            active = subprocess.run(['ssh',host,'systemctl','--user','is-active',C.NAME+'-'+family+'.service'],capture_output=True,text=True)
            if active.stdout.strip() in ('active','activating'):
                raise ValueError('ALREADY_RUNNING_DO_NOT_REDEPLOY')
            run(['ssh',host,shlex.join(['mkdir','-p',runtime,role['packet'],role['output']])])
            run(['rsync','-a',str(archive),host+':'+runtime+'/code.tar'])
            run(['ssh',host,shlex.join(['tar','-xf',runtime+'/code.tar','-C',runtime])])
            run(['rsync','-a','--checksum',spec['packet']+'/',host+':'+role['packet']+'/'])
            worker=dict(packet=role['packet'],output=role['output'],code=code,code_commit=commit,
                        **role.get('worker_options',{}))
            private = tmp/(family+'.json'); W.B.put(private,worker)
            remote_spec = role['output']+'/RUN.private.json'
            run(['rsync','-a',str(private),host+':'+remote_spec])
            wall=config.get('worker_wall_seconds',3600)
            module=config.get('module','tools.i6dmo.monitor_campaign')
            command=[role['launcher'],'-m','4352M','-t',str(wall)+'s','-n',C.NAME+'-'+family,'--',
                     role['python'],'-B','-m',module,'worker','--spec',remote_spec,'--family',family]
            unit=C.NAME+'-'+family
            (tmp/(unit+'.service')).write_text(service(command,runtime,env,memory='6G',timeout=wall+100))
            (tmp/(unit+'.timer')).write_text(timer())
            run(['ssh',host,'mkdir','-p','.config/systemd/user'])
            run(['rsync','-a',str(tmp/(unit+'.service')),str(tmp/(unit+'.timer')),host+':.config/systemd/user/'])
            run(['ssh',host,'systemctl','--user','daemon-reload'])
            # Separate synthetic tests must have passed on both runtimes before --deploy.
            run(['ssh',host,'systemctl','--user','enable','--now',unit+'.timer'])
        unit=C.NAME+'-collect'
        command=[config['coordinator_launcher'],'-m','1024M','-t','600s','-n',unit,'--',
            config['coordinator_python'],'-B','-m',config.get('module','tools.i6dmo.monitor_campaign'),
            'collect','--spec',str(root/'RUN.private.json')]
        folder=Path.home()/'.config/systemd/user';folder.mkdir(parents=True,exist_ok=True)
        env=dict(PYTHONPATH=str(W.REPO),CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        (folder/(unit+'.service')).write_text(service(command,str(W.REPO),env,memory='1536M',
            secret=config['warehouse']['environment_file'],timeout=650))
        (folder/(unit+'.timer')).write_text(timer())
        run(['systemctl','--user','daemon-reload'])
        run(['systemctl','--user','enable','--now',unit+'.timer'])
    return dict(state='DEPLOYED',commit=commit,units=len(doc['units']),arms=doc['arms'],fits=len(doc['units'])*len(doc['arms']))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True)
    args=p.parse_args()
    print(json.dumps(deploy(W.read(args.config)),sort_keys=True))


if __name__=='__main__':main()

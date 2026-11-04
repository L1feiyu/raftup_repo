"""Execute revision notebooks in the current Python environment with a real kernel.

Data remain external. Executed copies and reports are written only to --output-dir.
Example: python scripts/execute_revision_notebooks.py --mode tutorials
 --input-root /path/to/prepared_inputs --output-dir .validation/tutorial_run
"""
import argparse,json,os,sys,time,tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['tutorials','figures'],required=True)
    parser.add_argument('--input-root',type=Path)
    parser.add_argument('--archive-root',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--notebook',help='Optional filename prefix')
    parser.add_argument('--timeout',type=int,default=7200)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    source=repo/'examples/revision'
    if args.mode=='figures':source=source/'figures'
    if args.mode=='figures' and args.archive_root is None:parser.error('figures require --archive-root')
    names={'00':'smoke','01':'dlpfc_0708','02':'merfish_0409','03':'known_overlap','04':'deep_starmap_series','05':'ht206b1_u2_u3','06':'dlpfc_0708','07':'dlpfc_0708'}
    output=args.output_dir.expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLBACKEND','Agg')
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ.setdefault(key,'4')
    with tempfile.TemporaryDirectory(prefix='raftup-kernel-') as temp:
        directory=Path(temp)/'kernels/raftup_validation';directory.mkdir(parents=True)
        (directory/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'RAFT-UP validation','language':'python'}))
        os.environ['JUPYTER_PATH']=temp+os.pathsep+os.environ.get('JUPYTER_PATH','')
        for path in sorted(source.glob('*.ipynb')):
            if args.notebook and not path.name.startswith(args.notebook):continue
            dest=output/path.stem
            if (dest/'result.json').exists():raise FileExistsError(f'Use a fresh output directory: {dest}')
            dest.mkdir(parents=True,exist_ok=True)
            os.environ['RAFTUP_OUTPUT_ROOT']=str(dest/'outputs')
            if args.mode=='figures':os.environ['RAFTUP_EXPERIMENT_ROOT']=str(args.archive_root.resolve())
            elif args.input_root:os.environ['RAFTUP_TUTORIAL_DATA']=str(args.input_root.resolve()/names[path.name[:2]])
            print('Running',path.name,flush=True)
            nb=nbformat.read(path,as_version=4);start=time.time();error=None
            def report_cell(cell,cell_index,**kwargs):print(f'  cell {cell_index}',flush=True)
            try:
                NotebookClient(nb,timeout=args.timeout,kernel_name='raftup_validation',
                    on_cell_start=report_cell,resources={'metadata':{'path':str(source)}}).execute()
            except Exception as exc:error=str(exc)
            nbformat.write(nb,dest/'executed.ipynb')
            result=dict(notebook=path.name,status='failed' if error else 'passed',seconds=time.time()-start,
                        python=sys.executable,error=error)
            (dest/'result.json').write_text(json.dumps(result,indent=2))
            print(result,flush=True)
            if error:raise RuntimeError(error)
if __name__=='__main__':main()

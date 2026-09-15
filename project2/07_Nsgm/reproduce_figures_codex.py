"""Regenerate every manuscript figure from maintained sources, without publishing.

Run with the repository Python and --paper <paper directory>. Existing fit caches
are reused. Each run keeps a paper backup, fresh figures, and a pixel comparison
under __codex_ignore. Notebook order preserves the B64 analysis dependencies.
Topology sources stay in the OneDrive diagrams_Nsgm folder beside the paper.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
GROUPS = {
    'cB211.072.64/analysis_2pt_codex': [
        'C2pt_N_GEVP_compare', 'C2pt_Nsgm_GEVP_compare', 'C2pt_overlap_compare',
        'GEVP_vw', 'C2pt_laplace_delta_dependence'],
    'cB211.072.64/analysis_3pt_light_codex': [
        'Rstd_RGEVP', 'caseIII_excited_matrix_element', 'Rd_compare_w_v2',
        'Rstd_RLap', 'energy_scales', 'RLG_RL2G', 'RLap_delta_dependence'],
    'cB211.072.64/analysis_3pt_strange_charm_codex': [
        'sigma_s_W_vs_full', 'sigma_s_Rstd_RGEVP', 'sigma_s_Rstd_RLap', 'sigma_c_W_vs_full', 'sigma_c_Rstd_RGEVP'],
    'cA211.530.24/analysis_2pt_codex': [
        'C2pt_N_GEVP_compare', 'C2pt_Nsgm_GEVP_compare', 'C2pt_overlap_compare',
        'GEVP_vw'],
    'cA211.530.24/analysis_3pt_light_codex': [
        'Rstd_RGEVP_light_A24', 'Laplace_summary_light_A24', 'energy_scales'],
    'cA2.09.48/analysis_2pt_codex': [
        'C2pt_N_GEVP_compare', 'C2pt_Nsgm_GEVP_compare', 'C2pt_overlap_compare',
        'GEVP_vw'],
    'cA2.09.48/analysis_3pt_light_codex': [
        'Rstd_RGEVP_light_A48', 'Laplace_summary_light_A48', 'energy_scales'],
    'analysis_3pt_topologies_codex': ['gevp_midpoint_differences'],
}
DIAGRAMS = {
    'diags_2pt': 'diags_2pt_codex.tex',
    'diags_3pt_Id': 'diags_3pt_Id_codex.tex',
    'diags_3pt_NsgmJNsgm_codex': 'diags_3pt_NsgmJNsgm_codex.tex',
}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publication_name(group, name):
    suffix = {'cA211.530.24': 'A24', 'cA2.09.48': 'A48'}.get(str(Path(group).parent))
    return name + ('_' + suffix if suffix and not name.endswith('_' + suffix) else '') + '.pdf'


def main(paper, diagrams=None):
    diagrams = paper.parent / 'diagrams_Nsgm' if diagrams is None else diagrams
    work = ROOT / '__codex_ignore' / datetime.now().strftime('regenerate_paper_%Y%m%d_%H%M%S')
    work.mkdir()
    print(f'RUN DIRECTORY: {work}', flush=True)
    for key in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
        os.environ[key] = '1'
    for key in ['JUPYTER_CONFIG_DIR', 'JUPYTER_RUNTIME_DIR', 'IPYTHONDIR']:
        os.environ[key] = str(work / 'runtime')

    import nbformat
    from nbclient import NotebookClient
    from PIL import Image, ImageChops

    figures = {}
    for group, names in GROUPS.items():
        ensemble, notebook = str(Path(group).parent), Path(group).name
        for name in names:
            figures[publication_name(group, name)] = {
                'source': ROOT / (group + '.ipynb'),
                'output': ROOT / '__codex_ignore' / ensemble / 'fig' / notebook / 'internal_ignore' / (name + '.pdf'),
            }
    for name, source in DIAGRAMS.items():
        figures[name + '.pdf'] = {'source': diagrams / source,
                                'output': work / 'diagrams' / (name + '.pdf')}

    text = (paper / 'main.tex').read_text(encoding='utf-8')
    included = re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text)
    if set(included) != set(figures):
        raise ValueError(f'Figure-map mismatch: {set(included) ^ set(figures)}')
    sources = {item['source'] for item in figures.values()}
    sources.update(ROOT / name for name in ['util.py', 'util_codex.py', 'util_Nsgm.py'])
    sources.update(ROOT / ensemble / 'processData_codex.ipynb'
                   for ensemble in ['cA211.530.24', 'cA2.09.48', 'cB211.072.64'])
    for ensemble in ['cA211.530.24', 'cA2.09.48', 'cB211.072.64']:
        compact = ROOT / '__codex_ignore' / ensemble / 'pkl/processData_codex/reg_ignore/data_topologies.pkl'
        if not compact.is_file():
            raise FileNotFoundError(f'Run {ensemble}/processData_codex.ipynb once before reproduction: {compact}')
    sources.add(diagrams / 'nsgm_diagram_defs_codex.tex')
    for source in sources:
        assert source.is_file(), source
        if not source.is_relative_to(ROOT):
            continue
        check = subprocess.run(['git', 'check-ignore', '-q', str(source)], cwd=ROOT)
        if check.returncode != 1:
            raise RuntimeError(f'Source is ignored or Git check failed: {source}')

    before = work / 'before'
    before.mkdir()
    for name in ['main.tex', 'references.bib', 'main.pdf']:
        shutil.copy2(paper / name, before / name)
    shutil.copytree(paper / 'fig', before / 'fig')
    (work / 'previous_outputs').mkdir()
    for name, item in figures.items():
        if item['output'].exists():
            shutil.move(item['output'], work / 'previous_outputs' / name)

    for group in GROUPS:
        path = ROOT / (group + '.ipynb')
        notebook = nbformat.read(path, as_version=4)
        print(f'EXECUTE {group}', flush=True)
        client = NotebookClient(notebook, timeout=600, kernel_name='python3',
                                resources={'metadata': {'path': str(path.parent)}})
        try:
            client.execute()
        finally:
            executed = work / 'executed' / path.parent.name / path.name
            executed.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(notebook, executed)
        nbformat.write(notebook, path)
        print(f'FINISHED {group}', flush=True)

    (work / 'diagrams').mkdir()
    for name, source in DIAGRAMS.items():
        command = ['pdflatex', '-interaction=nonstopmode', '-halt-on-error',
                   f'-jobname={name}', f'-output-directory={work / "diagrams"}', source]
        with (work / 'diagrams' / (name + '-build.log')).open('w') as log:
            subprocess.run(command, cwd=diagrams, stdout=log, stderr=subprocess.STDOUT, check=True)

    (work / 'fig').mkdir()
    (work / 'render').mkdir()
    report = []
    for name in included:
        item = figures[name]
        assert item['output'].is_file(), f'Not regenerated: {name}'
        shutil.copy2(item['output'], work / 'fig' / name)
        images = []
        for tag, pdf in [('before', before / 'fig' / name), ('after', work / 'fig' / name)]:
            prefix = work / 'render' / (Path(name).stem + '_' + tag)
            subprocess.run(['pdftoppm', '-r', '120', '-singlefile', '-png', str(pdf), str(prefix)], check=True, capture_output=True)
            with Image.open(str(prefix) + '.png') as image:
                images.append(image.convert('RGB'))
        identical = images[0].size == images[1].size and ImageChops.difference(*images).getbbox() is None
        report.append({'figure': name, 'source': str(item['source']),
                       'pixels_identical': identical, 'before_sha256': file_hash(before / 'fig' / name),
                       'after_sha256': file_hash(work / 'fig' / name)})
        print(f'{name}: {"IDENTICAL" if identical else "DIFFERENT"}', flush=True)
    (work / 'figure_report.json').write_text(json.dumps(report, indent=2) + '\n')
    source_hashes = {str(path): file_hash(path) for path in sources}
    (work / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
    print(f'Regenerated {len(report)} figures. Pixel-identical: {sum(row["pixels_identical"] for row in report)}.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, required=True)
    parser.add_argument('--diagrams', type=Path, help='Original topology-source directory when using a staged manuscript')
    arguments = parser.parse_args()
    main(arguments.paper.resolve(), arguments.diagrams.resolve() if arguments.diagrams else None)

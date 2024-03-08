from pathlib import Path
import subprocess
import os
import urllib.request

def system_command(command):
    """Execute a system command with subprocess."""
    subprocess.run(command, shell=True, check=True)

def download_file(url, local_filename=None):
    """Download a file from a URL using urllib."""
    if not local_filename:
        local_filename = url.split('/')[-1]
    with urllib.request.urlopen(url) as response, open(local_filename, 'wb') as out_file:
        out_file.write(response.read())
    return local_filename

def compile_kenlm(max_order=12):
    """Compile KenLM with the specified maximum order."""
    url = "https://kheafield.com/code/kenlm.tar.gz"
    kenlm_tar = download_file(url)

    # Extract KenLM archive
    system_command(f'tar -xvzf {kenlm_tar}')

    # Setup KenLM directory paths using pathlib
    kenlm_dir = Path('kenlm')
    build_dir = kenlm_dir / 'build'
    build_dir.mkdir(parents=True, exist_ok=True)

    # Compile KenLM
    os.chdir(build_dir)
    system_command(f'cmake .. -DKENLM_MAX_ORDER={max_order}')
    system_command('make -j 4')
    os.chdir('../../')

    # Clean up downloaded files
    Path(kenlm_tar).unlink()

if __name__ == "__main__":
    compile_kenlm(max_order=12)
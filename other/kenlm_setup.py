import logging
from pathlib import Path
import subprocess
import os
import urllib.request

# Configure basic logging
logging.basicConfig(level=logging.INFO)

def system_command(command_list, cwd=None):
    """
    Execute a system command in a specified directory.

    This function uses subprocess to execute a given command directly, bypassing the shell to enhance security.
    It optionally allows execution in a specified directory.

    Parameters:
    - command_list (list): The command and its arguments to execute as a list.
    - cwd (Path or str, optional): Directory to execute the command in. Defaults to None.

    Raises:
    - subprocess.CalledProcessError: If the command execution fails.
    """
    try:
        subprocess.run(command_list, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        logging.error(f"Command {command_list} failed with error: {e}")
        raise

def download_file(url, local_filename=None):
    """
    Download a file from a specified URL.

    If a local filename is not provided, it deduces the filename from the URL.
    Writes the file in binary mode to the local system.

    Parameters:
    - url (str): The URL of the file to download.
    - local_filename (Path or str, optional): The local file path to save the downloaded file. Defaults to None.

    Returns:
    - Path: The path to the downloaded file.

    Raises:
    - Exception: For any error that occurs during the download process.
    """
    try:
        if not local_filename:
            local_filename = Path(url.split('/')[-1])
        
        with urllib.request.urlopen(url) as response, open(local_filename, 'wb') as out_file:
            out_file.write(response.read())
        return local_filename
    except Exception as e:
        logging.error(f"Failed to download {url} with error: {e}")
        raise

def compile_kenlm(max_order=12):
    """
    Compile KenLM with the specified maximum order.

    This function automates the process of downloading, extracting, and compiling the KenLM library from its source.
    It sets the maximum order for n-grams during compilation and modifies the script's execution environment PATH
    to include the compiled binaries, making them immediately usable.

    Parameters:
    - max_order (int, optional): The maximum order of n-grams. Defaults to 12.
    """
    # Download the KenLM source code
    url = "https://kheafield.com/code/kenlm.tar.gz"
    kenlm_tar = download_file(url)
    
    # Extract the KenLM archive
    system_command(['tar', '-xvzf', str(kenlm_tar)])
    
    # Prepare the KenLM build directory
    kenlm_dir = Path.cwd() / 'kenlm'
    build_dir = kenlm_dir / 'build'
    build_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile KenLM using cmake and make, within the build directory
    system_command(['cmake', '..', f'-DKENLM_MAX_ORDER={max_order}'], cwd=build_dir)
    system_command(['make', '-j4'], cwd=build_dir)
    
    # Append the KenLM binaries directory to PATH for immediate use
    bin_dir = build_dir / 'bin'
    os.environ["PATH"] += os.pathsep + str(bin_dir)
    
    # Cleanup: remove the downloaded KenLM archive
    kenlm_tar.unlink()

if __name__ == "__main__":
    compile_kenlm(max_order=12)

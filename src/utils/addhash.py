import subprocess
import os

def is_git_directory(path = '.'):
    return subprocess.call(['git', '-C', path, 'status'], stderr=subprocess.STDOUT, stdout = open(os.devnull, 'w')) == 0

def find_git_root(path='.') -> str | None:
    """Find the git root directory from a given path by walking up the directory tree."""
    path = os.path.abspath(path)
    
    # If path is a file, get its directory
    if os.path.isfile(path):
        path = os.path.dirname(path)
    
    # Walk up the directory tree looking for .git
    current = path
    while current != '/':
        if os.path.isdir(os.path.join(current, '.git')):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # reached root
            break
        current = parent
    
    return None

def get_git_revision_short_hash(path='.') -> str:
    if is_git_directory(path):
        return subprocess.check_output(['git', '-C', path, 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    else: return 'Not run locally.'

def get_git_revision_hash(path='.') -> str:
    if is_git_directory(path): 
        return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD']).decode('ascii').strip()
    else: return 'Not run locally.'


def get_git_diff_master(path='.') -> str:
    if is_git_directory(path): 
        return subprocess.check_output(['git', '-C', path, 'diff', 'origin/master', 'HEAD']).decode('ascii')
    else: return 'Not run locally.'

def get_git_diff(path='.') -> str:
    if is_git_directory(path):
        # Get the list of files not ignored
        files = subprocess.check_output(['git', '-C', path, 'ls-files']).decode('ascii').strip().split('\n')
        if files:
            # Filter to only existing files to avoid git diff errors on moved/deleted files
            existing_files = [f for f in files if os.path.exists(os.path.join(path, f))]
            if existing_files:
                # Pass these files to git diff
                try:
                    return subprocess.check_output(['git', '-C', path, 'diff', 'HEAD'] + existing_files).decode('utf-8')
                except UnicodeDecodeError:
                    # Fallback for binary files or encoding issues
                    return subprocess.check_output(['git', '-C', path, 'diff', 'HEAD'] + existing_files).decode('utf-8', errors='ignore')
            else:
                return 'No tracked files exist to diff.'
        else:
            return 'No changes to show.'
    else:
        return 'Not run locally'

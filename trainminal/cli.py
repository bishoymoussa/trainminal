"""CLI interface for trainminal."""

import os
import sys
import subprocess
import signal
import click
import threading
from pathlib import Path
from typing import Optional

from trainminal.monitor import Monitor
from trainminal.metrics import Phase


# Global variable to store command history
_command_history_file = Path.home() / ".trainminal_history"
_last_command: Optional[str] = None


def _save_command(command: str):
    """Save command to history file."""
    global _last_command
    _last_command = command
    
    try:
        with open(_command_history_file, 'a') as f:
            f.write(f"{command}\n")
    except Exception:
        pass


def _load_last_command() -> Optional[str]:
    """Load the last command from history."""
    global _last_command
    
    if _last_command:
        return _last_command
    
    try:
        if _command_history_file.exists():
            with open(_command_history_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    return lines[-1].strip()
    except Exception:
        pass
    
    return None


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Trainminal - Terminal ML Training Monitor."""
    pass


@main.command()
@click.argument('command', nargs=-1, required=True)
@click.option('--no-tui', is_flag=True, help='Disable TUI mode')
@click.option('--refresh-rate', default=0.5, help='Display refresh rate in seconds')
@click.option('--log-file', help='Log metrics to file')
def run(command, no_tui, refresh_rate, log_file):
    """Run a training script with monitoring.
    
    Example: trainminal run python train.py --epochs 10
    """
    cmd_str = ' '.join(command)
    _save_command(cmd_str)
    
    # Create a shared state file for metrics
    import tempfile
    shared_state_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
    shared_state_file.close()
    shared_state_path = shared_state_file.name
    
    # Set environment variable so script's monitor knows to use shared state
    env = os.environ.copy()
    env['TRAINMINAL_SHARED_STATE'] = shared_state_path
    env['TRAINMINAL_CLI_MODE'] = '1'
    
    # Create monitor
    monitor = Monitor(use_tui=not no_tui, refresh_rate=refresh_rate, log_file=log_file)
    
    try:
        with monitor:
            # Set phase to indicate process is running
            from trainminal.metrics import Phase
            monitor.set_phase(Phase.TRAINING)
            
            # Run the command
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=env
            )
            
            # Monitor the process and read shared state
            try:
                import json
                import time
                
                # Thread to read shared state file
                def read_shared_state():
                    while process.poll() is None:
                        try:
                            if os.path.exists(shared_state_path):
                                with open(shared_state_path, 'r') as f:
                                    try:
                                        state = json.load(f)
                                        # Update monitor with metrics from shared state
                                        if 'metrics' in state:
                                            for name, value in state['metrics'].items():
                                                monitor.log_metric(name, value)
                                        if 'phase' in state:
                                            monitor.set_phase(Phase[state['phase']])
                                        if 'epoch' in state:
                                            monitor.set_epoch(state.get('epoch', 0), 
                                                            state.get('total_epochs'))
                                        if 'batch' in state:
                                            monitor.set_batch(state.get('batch', 0),
                                                            state.get('total_batches'))
                                        if 'learning_rate' in state:
                                            monitor.set_learning_rate(state['learning_rate'])
                                    except (json.JSONDecodeError, KeyError):
                                        pass
                        except Exception:
                            pass
                        time.sleep(0.1)  # Check every 100ms
                
                state_thread = threading.Thread(target=read_shared_state, daemon=True)
                state_thread.start()
                
                # Read process output (but don't block on it)
                import select
                import queue
                output_queue = queue.Queue()
                
                def read_output():
                    for line in process.stdout:
                        output_queue.put(line)
                
                output_thread = threading.Thread(target=read_output, daemon=True)
                output_thread.start()
                
                # Wait for process to complete
                exit_code = process.wait()
                if exit_code != 0:
                    # Process exited with error, but exception handling
                    # should have been done by the script itself
                    pass
            except KeyboardInterrupt:
                click.echo("\n[!] Interrupted by user")
                process.send_signal(signal.SIGINT)
                process.wait()
                raise
    except Exception as e:
        # If monitor itself has an exception, handle it
        if monitor.exception_handler:
            monitor.exception_handler.display_exception(
                type(e), e, e.__traceback__
            )
        click.echo(f"[!] Error: {e}", err=True)
        sys.exit(1)
    finally:
        # Clean up shared state file
        try:
            if 'shared_state_path' in locals() and os.path.exists(shared_state_path):
                os.unlink(shared_state_path)
        except Exception:
            pass


@main.command()
@click.option('--no-tui', is_flag=True, help='Disable TUI mode')
@click.option('--refresh-rate', default=0.5, help='Display refresh rate in seconds')
@click.option('--log-file', help='Log metrics to file')
def redo(no_tui, refresh_rate, log_file):
    """Redo the last command that was run with trainminal."""
    last_cmd = _load_last_command()
    
    if not last_cmd:
        click.echo("[!] No previous command found in history", err=True)
        sys.exit(1)
    
    click.echo(f"[*] Running: {last_cmd}")
    
    # Split command and run
    import shlex
    cmd_parts = shlex.split(last_cmd)
    
    # Re-run with trainminal by calling run function
    run.callback(cmd_parts, no_tui, refresh_rate, log_file)


@main.command()
@click.option('--pid', type=int, help='Process ID to kill')
def kill(pid):
    """Kill a training process.
    
    If PID is not provided, attempts to find and kill the last trainminal process.
    """
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            click.echo(f"[*] Sent SIGTERM to process {pid}")
        except ProcessLookupError:
            click.echo(f"[!] Process {pid} not found", err=True)
            sys.exit(1)
        except PermissionError:
            click.echo(f"[!] Permission denied to kill process {pid}", err=True)
            sys.exit(1)
    else:
        # Try to find the last trainminal process
        try:
            import psutil
            current_pid = os.getpid()
            
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if proc.info['pid'] == current_pid:
                        continue
                    
                    cmdline = proc.info.get('cmdline', [])
                    if cmdline and 'trainminal' in ' '.join(cmdline):
                        proc.terminate()
                        click.echo(f"[*] Terminated trainminal process {proc.info['pid']}")
                        return
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            click.echo("[!] No trainminal process found", err=True)
        except ImportError:
            click.echo("[!] psutil not available. Please provide PID explicitly.", err=True)
            sys.exit(1)


@main.command()
def history():
    """Show command history."""
    try:
        if _command_history_file.exists():
            with open(_command_history_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    click.echo("[*] Command History:")
                    for i, line in enumerate(lines[-10:], 1):  # Show last 10
                        click.echo(f"  {i}. {line.strip()}")
                else:
                    click.echo("[*] No command history")
        else:
            click.echo("[*] No command history")
    except Exception as e:
        click.echo(f"[!] Error reading history: {e}", err=True)


if __name__ == '__main__':
    main()


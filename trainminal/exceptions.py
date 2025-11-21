"""Exception handling and display for trainminal."""

import sys
import traceback
from typing import Optional, Type, Tuple
from io import StringIO
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.syntax import Syntax


class ExceptionHandler:
    """Handle and display exceptions cleanly."""
    
    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.last_exception: Optional[Exception] = None
        self.last_traceback: Optional[str] = None
    
    def format_exception(self, exc_type: Type[Exception], exc_value: Exception, 
                        exc_traceback) -> Tuple[str, str]:
        """Format exception into clean, copy-pastable text."""
        # Get the full traceback
        tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        full_traceback = ''.join(tb_lines)
        
        # Extract just the exception message (last line)
        exception_msg = str(exc_value)
        
        # Get the traceback without the exception message line
        traceback_only = '\n'.join(tb_lines[:-1])
        
        return exception_msg, full_traceback
    
    def display_exception(self, exc_type: Type[Exception], exc_value: Exception,
                         exc_traceback, show_full_traceback: bool = True):
        """Display exception in a clean, copy-pastable format."""
        exception_msg, full_traceback = self.format_exception(exc_type, exc_value, exc_traceback)
        
        # Store for later access
        self.last_exception = exc_value
        self.last_traceback = full_traceback
        
        # Stop any live displays first and clear screen
        try:
            self.console.clear_live()
        except:
            pass
        
        # Clear the screen for clean exception display
        import sys
        sys.stdout.write("\033[2J\033[H")  # Clear screen and move cursor to top
        sys.stdout.flush()
        
        # Create clean exception display
        error_text = Text()
        error_text.append("EXCEPTION OCCURRED\n\n", style="bold red")
        error_text.append(f"Type: {exc_type.__name__}\n", style="yellow")
        error_text.append(f"Message: {exception_msg}\n\n", style="white")
        
        if show_full_traceback:
            # Format traceback with syntax highlighting
            syntax = Syntax(
                full_traceback,
                "python",
                theme="monokai",
                line_numbers=True,
                word_wrap=True
            )
            
            # Display in a panel
            header_panel = Panel(
                error_text,
                title="[red]Exception[/red]",
                border_style="red",
                expand=False
            )
            
            traceback_panel = Panel(
                syntax,
                title="[red]Full Traceback[/red]",
                border_style="red",
                expand=False
            )
            
            self.console.print()
            self.console.print(header_panel)
            self.console.print()
            self.console.print(traceback_panel)
        else:
            # Just show the exception message
            panel = Panel(
                error_text,
                title="[red]Exception[/red]",
                border_style="red",
                expand=False
            )
            self.console.print()
            self.console.print(panel)
        
        # Print clean copy-pastable version (plain text, no formatting)
        print("\n" + "=" * 80, file=sys.stderr)
        print("COPY-PASTABLE EXCEPTION TEXT:", file=sys.stderr)
        print("=" * 80, file=sys.stderr)
        print(full_traceback, file=sys.stderr, end='')
        print("=" * 80 + "\n", file=sys.stderr)
    
    def get_exception_text(self) -> Optional[str]:
        """Get the last exception as plain text for copying."""
        return self.last_traceback
    
    def save_exception_to_file(self, filepath: str):
        """Save exception to a file."""
        if self.last_traceback:
            with open(filepath, 'w') as f:
                f.write(self.last_traceback)
            return True
        return False


def format_exception_clean(exc_type: Type[Exception], exc_value: Exception,
                           exc_traceback) -> str:
    """Format exception as clean text without any formatting."""
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    return ''.join(tb_lines)


def print_exception_clean(exc_type: Type[Exception], exc_value: Exception,
                          exc_traceback, file=None):
    """Print exception in clean format to a file (default: stderr)."""
    if file is None:
        file = sys.stderr
    
    exception_text = format_exception_clean(exc_type, exc_value, exc_traceback)
    
    # Clear any potential interference
    print("\n" + "=" * 80, file=file)
    print("EXCEPTION:", file=file)
    print("=" * 80, file=file)
    print(exception_text, file=file, end='')
    print("=" * 80 + "\n", file=file)


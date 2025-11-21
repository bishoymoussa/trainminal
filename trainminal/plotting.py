"""ASCII plotting for metrics visualization."""

from typing import Dict, List, Optional, Tuple
from collections import deque
import math


class ASCIIPlotter:
    """Create ASCII plots for metrics."""
    
    def __init__(self, width: int = 50, height: int = 10, max_points: int = 100):
        self.width = width
        self.height = height
        self.max_points = max_points
        self.metric_history: Dict[str, deque] = {}
    
    def add_point(self, metric_name: str, value: float):
        """Add a data point for a metric."""
        if metric_name not in self.metric_history:
            self.metric_history[metric_name] = deque(maxlen=self.max_points)
        self.metric_history[metric_name].append(value)
    
    def plot(self, metric_name: str, title: Optional[str] = None) -> str:
        """Create an ASCII plot for a metric."""
        if metric_name not in self.metric_history or len(self.metric_history[metric_name]) == 0:
            return f"No data for {metric_name}"
        
        data = list(self.metric_history[metric_name])
        if len(data) == 0:
            return f"No data for {metric_name}"
        
        # Need at least 2 points for a meaningful plot
        if len(data) < 2:
            latest = data[-1]
            title_line = f"{metric_name}: {latest:.6f}" if not title else title
            return f"{title_line}\n(Need 2+ points to plot)"
        
        # Calculate min/max for scaling
        min_val = min(data)
        max_val = max(data)
        
        # Handle edge case where all values are the same
        if max_val == min_val:
            # Show a flat line
            line = "─" * self.width
            title_line = f"{metric_name}: {max_val:.6f}" if not title else title
            return f"{title_line}\n{line}"
        
        # Create the plot grid
        plot_lines = []
        
        # Add title
        if title:
            plot_lines.append(title)
        else:
            plot_lines.append(f"{metric_name} (min: {min_val:.4f}, max: {max_val:.4f})")
        
        # Create the plot
        for row in range(self.height):
            y_val = max_val - (row / (self.height - 1)) * (max_val - min_val)
            line_chars = [' '] * self.width
            
            # Plot points
            for i, value in enumerate(data):
                x_pos = int((i / (len(data) - 1)) * (self.width - 1)) if len(data) > 1 else 0
                x_pos = min(x_pos, self.width - 1)
                
                # Calculate y position
                y_pos = int(((max_val - value) / (max_val - min_val)) * (self.height - 1))
                y_pos = max(0, min(y_pos, self.height - 1))
                
                if y_pos == row:
                    # Use different characters for different positions
                    if i == len(data) - 1:
                        line_chars[x_pos] = '●'  # Latest point
                    elif i == 0:
                        line_chars[x_pos] = '○'  # First point
                    else:
                        line_chars[x_pos] = '·'  # Middle points
            
            # Draw connecting lines
            if len(data) > 1:
                for i in range(len(data) - 1):
                    x1 = int((i / (len(data) - 1)) * (self.width - 1))
                    x2 = int(((i + 1) / (len(data) - 1)) * (self.width - 1))
                    x1 = min(x1, self.width - 1)
                    x2 = min(x2, self.width - 1)
                    
                    y1 = int(((max_val - data[i]) / (max_val - min_val)) * (self.height - 1))
                    y2 = int(((max_val - data[i + 1]) / (max_val - min_val)) * (self.height - 1))
                    y1 = max(0, min(y1, self.height - 1))
                    y2 = max(0, min(y2, self.height - 1))
                    
                    if row == y1 == y2:
                        # Horizontal line
                        for x in range(min(x1, x2), max(x1, x2) + 1):
                            if line_chars[x] == ' ':
                                line_chars[x] = '─'
                    elif x1 == x2:
                        # Vertical line
                        if min(y1, y2) <= row <= max(y1, y2):
                            if line_chars[x1] == ' ':
                                line_chars[x1] = '│'
                    else:
                        # Diagonal line approximation
                        if min(y1, y2) <= row <= max(y1, y2):
                            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                            x_at_row = x1 + (row - y1) / slope if slope != 0 else x1
                            x_at_row = int(x_at_row)
                            if 0 <= x_at_row < self.width:
                                if line_chars[x_at_row] == ' ':
                                    line_chars[x_at_row] = '│' if abs(slope) > 1 else '─'
            
            plot_lines.append(''.join(line_chars))
        
        # Add x-axis labels
        if len(data) > 1:
            x_labels = [' '] * self.width
            # Mark start and end
            x_labels[0] = '0'
            x_labels[-1] = str(len(data) - 1)
            plot_lines.append(''.join(x_labels))
        
        return '\n'.join(plot_lines)
    
    def plot_multiple(self, metric_names: List[str], title: Optional[str] = None) -> str:
        """Create multiple plots side by side."""
        plots = []
        for metric_name in metric_names:
            plots.append(self.plot(metric_name))
        
        if not plots:
            return "No metrics to plot"
        
        # Combine plots horizontally
        plot_lines_list = [p.split('\n') for p in plots]
        max_lines = max(len(lines) for lines in plot_lines_list)
        
        # Pad all plots to same height
        for lines in plot_lines_list:
            while len(lines) < max_lines:
                lines.append(' ' * self.width)
        
        # Combine horizontally
        combined = []
        for i in range(max_lines):
            line_parts = [lines[i] if i < len(lines) else ' ' * self.width 
                         for lines in plot_lines_list]
            combined.append('  │  '.join(line_parts))
        
        if title:
            return title + '\n' + '\n'.join(combined)
        return '\n'.join(combined)
    
    def clear(self, metric_name: Optional[str] = None):
        """Clear history for a metric or all metrics."""
        if metric_name:
            if metric_name in self.metric_history:
                self.metric_history[metric_name].clear()
        else:
            self.metric_history.clear()


def create_simple_plot(data: List[float], width: int = 50, height: int = 10) -> str:
    """Create a simple ASCII plot from a list of values."""
    plotter = ASCIIPlotter(width=width, height=height)
    for value in data:
        plotter.add_point("value", value)
    return plotter.plot("value")


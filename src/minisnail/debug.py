from rich.console import Console
import matplotlib.pyplot as plt
import time
import numpy as np

DEBUG = True
console = Console()

def plot_loss_curve_basic(loss_list, title="Training Loss Curve", save_path=None):
    """
    基础版本 - 绘制loss曲线
    
    参数:
    loss_list: 包含loss值的列表
    title: 图表标题
    save_path: 图片保存路径（可选）
    """
    plt.figure(figsize=(10, 6))
    
    # 绘制loss曲线
    plt.plot(range(1, len(loss_list) + 1), loss_list, 
             color='royalblue', linewidth=2, label='Loss')
    
    # 添加标记点
    min_loss_idx = np.argmin(loss_list)
    plt.scatter(min_loss_idx + 1, loss_list[min_loss_idx], 
                color='red', zorder=5, label=f'Min Loss: {loss_list[min_loss_idx]:.4f}')
    
    # 设置图表属性
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存至: {save_path}")
    
    plt.show()

class LossMonitor:
    """Loss Monitor"""
    
    def __init__(self, title: str ="Loss Monitor", show_stats: bool = True, window_size=10, update_interval=0.1):
        # Start interactive mode
        if show_stats:
            plt.ion()
        
            # Create figure and subplots
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Initialize data
        self.epochs = []
        self.losses = []
        self.moving_avg = []
        self.title = title
        self.show_stats = show_stats
        self.window_size = window_size
        
        # Precompute statistics variables to avoid redundant calculations
        self.min_loss = float('inf')
        self.max_loss = float('-inf')
        self.sum_loss = 0.0
        self.sum_squared_loss = 0.0
        self.count = 0
        self.first_loss = None
        
        if show_stats:
            # Create line plots
            self.loss_line, = self.ax1.plot([], [], 'b-', linewidth=1.5, alpha=0.7, label='Loss')
            self.avg_line, = self.ax1.plot([], [], 'r-', linewidth=2, label=f'{window_size}-Epoch MA')
            
            # Create scatter plot for recent points
            self.recent_scatter = self.ax1.scatter([], [], color='green', s=50, zorder=5, label='Recent')
            
            # Create text statistics
            self.stats_text = self.ax2.text(0.05, 0.95, '', transform=self.ax2.transAxes, 
                                            verticalalignment='top', fontsize=10,
                                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Set plot properties
            self.ax1.set_title(title, fontsize=14, fontweight='bold')
            self.ax1.set_xlabel('Epoch')
            self.ax1.set_ylabel('Loss')
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend(loc='upper right')
            
            # Hide second axis (used for text display)
            self.ax2.axis('off')
        
            # Set update interval
            self.update_interval = update_interval
            self.last_update_time = time.time()
            
            plt.tight_layout()
    
    def _calculate_moving_average(self):
        """Calculate moving average"""
        if len(self.losses) >= self.window_size:
            window = self.losses[-self.window_size:]
            return np.mean(window)
        return None
    
    def _update_stats_text(self):
        """Update statistics text"""
        if self.count == 0:
            return
        
        avg_loss = self.sum_loss / self.count
        variance = (self.sum_squared_loss / self.count) - (avg_loss ** 2)
        std_loss = variance ** 0.5
        
        stats = [
            f"Current Epoch: {self.count}",
            f"Current Loss: {self.losses[-1]:.6f}",
            f"Min Loss: {self.min_loss:.6f}",
            f"Max Loss: {self.max_loss:.6f}",
            f"Average Loss: {avg_loss:.6f}",
            f"Loss Std: {std_loss:.6f}",
            f"Loss Rate: {100*(self.first_loss-self.losses[-1])/self.first_loss:.2f}%"
        ]
        
        if len(self.moving_avg) > 0:
            stats.append(f"Moving Average: {self.moving_avg[-1]:.6f}")
        
        self.stats_text.set_text('\n'.join(stats))
    
    def add_loss(self, epoch, loss) -> bool:
        """Add loss data, return if it's the minimum loss"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        # Update statistics variables in real time
        if self.first_loss is None:
            self.first_loss = loss
        self.sum_loss += loss
        self.sum_squared_loss += loss * loss
        self.count += 1
        
        is_min_loss = loss < self.min_loss
        if is_min_loss:
            self.min_loss = loss
        if loss > self.max_loss:
            self.max_loss = loss
        
        # Calculate moving average
        avg = self._calculate_moving_average()
        if avg is not None:
            self.moving_avg.append(avg)
        
        # Check if plot needs update
        if not self.show_stats:
            return is_min_loss
        
        # Update plot if time interval has passed
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self._update_plot()
            self.last_update_time = current_time
        
        return is_min_loss
    
    def _update_plot(self):
        """Update plot"""
        if len(self.epochs) == 0:
            return
        
        # Update loss curve
        self.loss_line.set_data(self.epochs, self.losses)
        
        # Update moving average curve
        if len(self.moving_avg) > 0:
            avg_epochs = self.epochs[self.window_size-1:]
            self.avg_line.set_data(avg_epochs, self.moving_avg)
        
        # Update recent points scatter plot (show last 5 points)
        recent_count = min(5, len(self.epochs))
        recent_epochs = self.epochs[-recent_count:]
        recent_losses = self.losses[-recent_count:]
        self.recent_scatter.set_offsets(np.column_stack([recent_epochs, recent_losses]))
        
        # Adjust axis limits
        if len(self.epochs) > 1:
            self.ax1.set_xlim(0, max(self.epochs) * 1.05)
            y_min = self.min_loss * 0.95
            y_max = self.max_loss * 1.05
            self.ax1.set_ylim(y_min, y_max)
        
        # Update statistics text
        self._update_stats_text()
        
        # Redraw plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def finalize(self, save_path="./data/model/loss_curve.png"):
        """Finalize training, show final plot"""
        # Save Loss plot
        plot_loss_curve_basic(self.losses, title=self.title, save_path=save_path)
        
        if not self.show_stats:
            return
        
        # Ensure the update is called one last time to show the final state
        self._update_plot()
        
        # Disable interactive mode
        plt.ioff()
        
        # Add final markers
        if len(self.losses) > 0:
            min_idx = np.argmin(self.losses)
            self.ax1.plot(self.epochs[min_idx], self.losses[min_idx], 'r*', 
                         markersize=15, label=f'Min Loss: {self.losses[min_idx]:.4f}')
            self.ax1.legend()

        plt.show()
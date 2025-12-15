import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

class PerformanceAnalyzer:
    @staticmethod
    def plot_fitness_progression(ca_results: Dict):
        fitness_history = ca_results.get('fitness_history', [])
        best_fitness_history = ca_results.get('best_fitness_history', [])
        if not fitness_history and not best_fitness_history:
            return None
        
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        generations = range(len(fitness_history) if fitness_history else len(best_fitness_history))
        
        if fitness_history:
            ax.plot(generations, fitness_history, label='Average Fitness', color='#3498db', 
                   alpha=0.6, linewidth=1.5, linestyle='--', marker='o', markersize=3,
                   markevery=max(1, len(fitness_history)//50))
        if best_fitness_history:
            ax.plot(generations, best_fitness_history, label='Best Fitness', color='#27ae60', 
                   linewidth=2.5, marker='s', markersize=4, markevery=max(1, len(best_fitness_history)//50))
            ax.fill_between(generations, best_fitness_history, alpha=0.2, color='#27ae60')
        
        ax.axhline(y=1.0, color='#e74c3c', linestyle=':', linewidth=2, alpha=0.7, label='Target (Solved)')
        
        if best_fitness_history and max(best_fitness_history) >= 1.0:
            solved_gen = next((i for i, f in enumerate(best_fitness_history) if f >= 1.0), None)
            if solved_gen is not None:
                ax.axvline(x=solved_gen, color='#e74c3c', linestyle='--', linewidth=1.5, alpha=0.6)
                ax.plot(solved_gen, 1.0, 'ro', markersize=10, zorder=5)
                ax.annotate(f'Solved at Gen {solved_gen}', xy=(solved_gen, 1.0), 
                           xytext=(solved_gen + len(generations)*0.05, 0.95),
                           arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
                           fontsize=10, fontweight='bold', color='#e74c3c')
        
        ax.set_xlabel('Generation', fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_ylabel('Fitness Score', fontsize=12, fontweight='bold', color='#2c3e50')
        ax.set_title('Cultural Algorithm Performance Across Generations', 
                    fontsize=14, fontweight='bold', color='#2c3e50', pad=20)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95, shadow=True, fancybox=True).get_frame().set_facecolor('white')
        
        if best_fitness_history:
            max_fit = max(best_fitness_history)
            ax.set_ylim(-0.05, min(1.05, max_fit * 1.05))
            final_fitness, max_fitness = best_fitness_history[-1], max_fit
            avg_final = fitness_history[-1] if fitness_history else None
            stats = f"Final Best: {final_fitness:.4f}\nMax Fitness: {max_fitness:.4f}\n"
            if avg_final:
                stats += f"Final Average: {avg_final:.4f}\n"
            stats += f"Generations: {len(best_fitness_history)}\nSolved: {'✓ Yes' if max_fitness >= 1.0 else '✗ No'}"
            ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='#2c3e50'),
                   family='monospace')
        ax.set_xlim(-len(generations)*0.02, len(generations)*1.02)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_comparison(results: Dict):
        fig = plt.figure(figsize=(12, 5), facecolor='white')
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], hspace=0.3, wspace=0.3)
        ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        bt, ca = results['backtracking'], results['cultural_algorithm']
        algorithms = ['Backtracking', 'Cultural\nAlgorithm']
        
        times = [bt['execution_time'], ca['execution_time']]
        bars = ax1.bar(algorithms, times, color=['#3498db', '#27ae60'], 
                      edgecolor='#2c3e50', linewidth=2, alpha=0.8)
        for bar, time_val in zip(bars, times):
            time_str = f'{time_val*1000:.2f}ms' if time_val < 0.001 else (f'{time_val*1000:.1f}ms' if time_val < 1 else f'{time_val:.3f}s')
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.05,
                    time_str, ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax1.set_ylabel('Execution Time', fontsize=11, fontweight='bold', color='#2c3e50')
        ax1.set_title('⚡ Performance Comparison', fontsize=12, fontweight='bold', color='#2c3e50', pad=15)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_axisbelow(True)
        
        success_rates = [int(bt['solved']), int(ca['solved'])]
        bars2 = ax2.bar(algorithms, success_rates, 
                       color=['#27ae60' if s else '#e74c3c' for s in success_rates],
                       edgecolor='#2c3e50', linewidth=2, alpha=0.8)
        for bar, label in zip(bars2, ['✓ Solved' if s else '✗ Failed' for s in success_rates]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, label,
                    ha='center', va='center', color='white', weight='bold', fontsize=11)
        ax2.set_ylabel('Success Status', fontsize=11, fontweight='bold', color='#2c3e50')
        ax2.set_title('🎯 Solution Success', fontsize=12, fontweight='bold', color='#2c3e50', pad=15)
        ax2.set_ylim(0, 1.2)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Failed', 'Solved'])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_axisbelow(True)
        
        metrics_text = f"Backtracking:\n  Steps: {bt.get('steps', 'N/A')}\n  Backtracks: {bt.get('backtracks', 'N/A')}\n\n"
        metrics_text += f"Cultural Algorithm:\n  Generations: {ca.get('generations', 'N/A')}\n  Best Fitness: {ca.get('best_fitness', 'N/A'):.4f}"
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=9, family='monospace',
                bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.8, edgecolor='#2c3e50', pad=5))
        plt.tight_layout(rect=[0, 0.1, 1, 0.98])
        return fig
    
    @staticmethod
    def print_detailed_comparison(results: Dict):
        bt, ca = results['backtracking'], results['cultural_algorithm']
        print("\n" + "="*50 + "\nDETAILED PERFORMANCE COMPARISON\n" + "="*50)
        print(f"\nBacktracking Algorithm:\n  ✓ Solved: {bt['solved']}\n  ⏱️  Time: {bt['execution_time']:.4f} seconds")
        print(f"  📊 Steps: {bt.get('steps', 'N/A')}\n  🔁 Backtracks: {bt.get('backtracks', 'N/A')}")
        print(f"\nCultural Algorithm:\n  ✓ Solved: {ca['solved']}\n  ⏱️  Time: {ca['execution_time']:.4f} seconds")
        print(f"  📈 Generations: {ca.get('generations', 'N/A')}\n  🎯 Best Fitness: {ca.get('best_fitness', 'N/A'):.4f}/{ca.get('max_fitness', 'N/A')}")
        winner = ("Backtracking (Faster)" if bt['execution_time'] < ca['execution_time'] else "Cultural Algorithm (Faster)") if bt['solved'] and ca['solved'] else \
                 ("Backtracking (Only reliable solver)" if bt['solved'] else "Cultural Algorithm (Only reliable solver)" if ca['solved'] else "No algorithm solved the puzzle")
        print(f"\n🏆 Result: {winner}")
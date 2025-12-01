import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

class PerformanceAnalyzer:
    """Handles performance analysis and visualization"""
    
    @staticmethod
    def plot_fitness_progression(ca_results: Dict):
        """Plot fitness progression for Cultural Algorithm"""
        if not ca_results.get('fitness_history'):
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(ca_results['fitness_history'], label='Average Fitness', alpha=0.7)
        plt.plot(ca_results['best_fitness_history'], label='Best Fitness', linewidth=2)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Cultural Algorithm Fitness Progression')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    @staticmethod
    def plot_comparison(results: Dict):
        """Create comparison plots between algorithms"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Execution time comparison
        algorithms = ['Backtracking', 'Cultural Algorithm']
        times = [
            results['backtracking']['execution_time'],
            results['cultural_algorithm']['execution_time']
        ]
        
        bars = ax1.bar(algorithms, times, color=['skyblue', 'lightcoral'])
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('Algorithm Performance Comparison')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time_val:.4f}s', ha='center', va='bottom')
        
        # Success comparison
        success_rates = [
            results['backtracking']['solved'],
            results['cultural_algorithm']['solved']
        ]
        success_labels = ['Solved' if solved else 'Failed' for solved in success_rates]
        
        colors = ['green' if solved else 'red' for solved in success_rates]
        bars2 = ax2.bar(algorithms, success_rates, color=colors)
        ax2.set_ylabel('Success')
        ax2.set_title('Solution Success')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, label in zip(bars2, success_labels):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    label, ha='center', va='center', color='white', weight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def print_detailed_comparison(results: Dict):
        """Print detailed comparison of results"""
        bt = results['backtracking']
        ca = results['cultural_algorithm']
        
        print("\n" + "="*50)
        print("DETAILED PERFORMANCE COMPARISON")
        print("="*50)
        
        print(f"\nBacktracking Algorithm:")
        print(f"  ✓ Solved: {bt['solved']}")
        print(f"  ⏱️  Time: {bt['execution_time']:.4f} seconds")
        print(f"  📊 Steps: {bt.get('steps', 'N/A')}")
        print(f"  🔁 Backtracks: {bt.get('backtracks', 'N/A')}")
        
        print(f"\nCultural Algorithm:")
        print(f"  ✓ Solved: {ca['solved']}")
        print(f"  ⏱️  Time: {ca['execution_time']:.4f} seconds")
        print(f"  📈 Generations: {ca.get('generations', 'N/A')}")
        print(f"  🎯 Best Fitness: {ca.get('best_fitness', 'N/A'):.4f}/{ca.get('max_fitness', 'N/A')}")
        
        # Determine winner
        if bt['solved'] and ca['solved']:
            if bt['execution_time'] < ca['execution_time']:
                winner = "Backtracking (Faster)"
            else:
                winner = "Cultural Algorithm (Faster)"
        elif bt['solved']:
            winner = "Backtracking (Only reliable solver)"
        elif ca['solved']:
            winner = "Cultural Algorithm (Only reliable solver)"
        else:
            winner = "No algorithm solved the puzzle"
        
        print(f"\n🏆 Result: {winner}")
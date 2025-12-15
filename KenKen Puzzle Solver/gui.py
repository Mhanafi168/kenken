import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from solvers.kenken_solver import KenKenSolver
from utils.performance import PerformanceAnalyzer

# --- Modern Color Palette ---
COLORS = {
    'bg': '#f4f6f9',           # Light Blue-Grey Background
    'card': '#ffffff',         # White container
    'accent': '#3498db',       # Primary Blue
    'accent_hover': '#2980b9', # Darker Blue
    'text': '#2c3e50',         # Dark Grey Text
    'subtext': '#7f8c8d',      # Light Grey Text
    'grid_line': '#ecf0f1',    # Very light grey lines
    'cage_border': '#2c3e50',  # Thick dark border for cages
    'selected': '#e8f6f3',     # Light Green/Blue for selection
    'success': '#27ae60',      # Green
    'error': '#c0392b'         # Red
}

class KenKenGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("KenKen Master - Intelligent Solver")
        self.root.geometry("1100x750")
        self.root.configure(bg=COLORS['bg'])
        
        # Core Logic
        self.solver = KenKenSolver()
        self.current_puzzle = None
        self.size_var = tk.IntVar(value=4)
        
        # Game State
        self.selected_cell = None
        self.user_grid = None
        self.cell_size = 0
        
        self.setup_styles()
        self.setup_layout()
        self.bind_events()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure Colors and Fonts
        style.configure('Main.TFrame', background=COLORS['bg'])
        style.configure('Card.TFrame', background=COLORS['card'], relief='flat')
        
        # Headings
        style.configure('Title.TLabel', background=COLORS['bg'], foreground=COLORS['text'], font=('Segoe UI', 18, 'bold'))
        style.configure('Subtitle.TLabel', background=COLORS['card'], foreground=COLORS['text'], font=('Segoe UI', 11, 'bold'))
        style.configure('Status.TLabel', background=COLORS['bg'], foreground=COLORS['subtext'], font=('Segoe UI', 10))
        
        # Buttons
        style.configure('Primary.TButton', font=('Segoe UI', 10, 'bold'), borderwidth=0, padding=10)
        style.map('Primary.TButton', background=[('active', COLORS['accent_hover']), ('!active', COLORS['accent'])], foreground=[('!active', 'white'), ('active', 'white')])

    def setup_layout(self):
        # Main Container with padding
        main_container = ttk.Frame(self.root, style='Main.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # --- Header ---
        header = ttk.Frame(main_container, style='Main.TFrame')
        header.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header, text="KenKen Solver", style='Title.TLabel').pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Ready to generate puzzle.")
        ttk.Label(header, textvariable=self.status_var, style='Status.TLabel').pack(side=tk.RIGHT, pady=5)
        
        # --- Content Area (Split View) ---
        content = ttk.Frame(main_container, style='Main.TFrame')
        content.pack(fill=tk.BOTH, expand=True)
        
        # Left Column: The Game Board
        self.left_panel = ttk.Frame(content, style='Card.TFrame', padding=15)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.canvas = tk.Canvas(self.left_panel, bg='white', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right Column: Controls & Stats
        self.right_panel = ttk.Frame(content, style='Main.TFrame', width=350)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_panel.pack_propagate(False) # Force width
        
        self.create_control_panel()
        self.create_log_panel()

    def create_control_panel(self):
        # Control Card
        card = ttk.Frame(self.right_panel, style='Card.TFrame', padding=15)
        card.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(card, text="Game Controls", style='Subtitle.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        # Size Selector
        row1 = ttk.Frame(card, style='Card.TFrame')
        row1.pack(fill=tk.X, pady=5)
        ttk.Label(row1, text="Grid Size:", background='white').pack(side=tk.LEFT)
        ttk.Spinbox(row1, from_=3, to=8, textvariable=self.size_var, width=5).pack(side=tk.RIGHT)
        
        # Main Actions
        ttk.Button(card, text="Generate New Puzzle", style='Primary.TButton', command=self.generate_puzzle).pack(fill=tk.X, pady=10)
        
        # Divider
        ttk.Separator(card, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Solver Actions
        ttk.Label(card, text="AI Solvers", style='Subtitle.TLabel').pack(anchor=tk.W, pady=(5, 5))
        
        self.btn_bt = ttk.Button(card, text="Solve (Backtracking)", command=lambda: self.run_threaded(self.solve_backtracking))
        self.btn_bt.pack(fill=tk.X, pady=2)
        
        self.btn_ca = ttk.Button(card, text="Solve (Cultural Algo)", command=lambda: self.run_threaded(self.solve_cultural))
        self.btn_ca.pack(fill=tk.X, pady=2)
        
        self.btn_cmp = ttk.Button(card, text="Compare Performance", command=lambda: self.run_threaded(self.compare_algorithms))
        self.btn_cmp.pack(fill=tk.X, pady=2)
        
        # Progress Bar (Hidden initially)
        self.progress = ttk.Progressbar(card, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=10)
        self.progress.pack_forget()

    def create_log_panel(self):
        # Logs Card
        card = ttk.Frame(self.right_panel, style='Card.TFrame', padding=15)
        card.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(card, text="Solution Logs", style='Subtitle.TLabel').pack(anchor=tk.W, pady=(0, 5))
        
        self.log_text = tk.Text(card, font=("Consolas", 9), height=10, relief='flat', bg='#f8f9fa')
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def bind_events(self):
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.root.bind("<Key>", self.on_key_press)
        # Handle window resize to redraw grid
        self.canvas.bind("<Configure>", lambda e: self.draw_board())

    # ==========================================
    # LOGIC
    # ==========================================

    def generate_puzzle(self):
        try:
            size = self.size_var.get()
            self.solver.generate_random_puzzle(size)
            self.current_puzzle = self.solver.puzzle
            self.user_grid = np.zeros((size, size), dtype=int)
            self.selected_cell = None
            
            self.log(f"\nGenerated {size}x{size} Puzzle.")
            self.log(f"Cages: {len(self.current_puzzle.cages)}")
            self.status_var.set("Puzzle generated. Click cells to play or use AI.")
            self.draw_board()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def draw_board(self):
        if not self.current_puzzle: return
        self.canvas.delete("all")
        
        # Dynamic sizing
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        size = self.current_puzzle.size
        self.cell_size = (min(w, h) - 40) // size
        start_x = (w - (self.cell_size * size)) // 2
        start_y = (h - (self.cell_size * size)) // 2
        cage_map = self._build_cage_map()
        
        for r in range(size):
            for c in range(size):
                x1, y1 = start_x + c * self.cell_size, start_y + r * self.cell_size
                x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, 
                    fill=COLORS['selected'] if (r, c) == self.selected_cell else 'white', 
                    outline=COLORS['grid_line'])
                cid = cage_map.get((r, c))
                if r == 0 or cage_map.get((r-1, c)) != cid:
                    self.canvas.create_line(x1, y1, x2, y1, width=3, fill=COLORS['cage_border'])
                if r == size-1 or cage_map.get((r+1, c)) != cid:
                    self.canvas.create_line(x1, y2, x2, y2, width=3, fill=COLORS['cage_border'])
                if c == 0 or cage_map.get((r, c-1)) != cid:
                    self.canvas.create_line(x1, y1, x1, y2, width=3, fill=COLORS['cage_border'])
                if c == size-1 or cage_map.get((r, c+1)) != cid:
                    self.canvas.create_line(x2, y1, x2, y2, width=3, fill=COLORS['cage_border'])
                if self.user_grid[r, c] != 0:
                    self.canvas.create_text(x1 + self.cell_size/2, y1 + self.cell_size/2, 
                        text=str(self.user_grid[r, c]), font=("Segoe UI", 18), fill=COLORS['text'])
        
        for cage in self.current_puzzle.cages:
            top_left = min(cage['cells'], key=lambda x: (x[0], x[1]))
            self.canvas.create_text(start_x + top_left[1] * self.cell_size + 5, 
                start_y + top_left[0] * self.cell_size + 5, 
                text=f"{cage['target']}{cage['operation']}", anchor="nw", 
                font=("Segoe UI", 9, "bold"), fill=COLORS['text'])
        
        self.board_offset = (start_x, start_y)

    def _build_cage_map(self):
        return {cell: i for i, cage in enumerate(self.current_puzzle.cages) for cell in cage['cells']}

    # ==========================================
    # INTERACTION
    # ==========================================

    def on_canvas_click(self, event):
        if not self.current_puzzle: return
        ox, oy = getattr(self, 'board_offset', (0, 0))
        r, c = int((event.y - oy) // self.cell_size), int((event.x - ox) // self.cell_size)
        if 0 <= r < self.current_puzzle.size and 0 <= c < self.current_puzzle.size:
            self.selected_cell = (r, c)
            self.draw_board()

    def on_key_press(self, event):
        if not self.selected_cell or not self.current_puzzle: return
        r, c = self.selected_cell
        if event.char.isdigit() and event.char != '0':
            val = int(event.char)
            if val <= self.current_puzzle.size:
                self.user_grid[r, c] = val
                self.draw_board()
        elif event.keysym in ('BackSpace', 'Delete'):
            self.user_grid[r, c] = 0
            self.draw_board()

    # ==========================================
    # THREADING & SOLVERS
    # ==========================================

    def run_threaded(self, func):
        """Prevents GUI Freeze"""
        if not self.current_puzzle: 
            messagebox.showwarning("Oops", "Generate a puzzle first!")
            return
            
        self.toggle_ui(False)
        self.progress.start(10)
        self.status_var.set("Working...")
        
        threading.Thread(target=func, daemon=True).start()

    def toggle_ui(self, enable):
        state = tk.NORMAL if enable else tk.DISABLED
        for btn in [self.btn_bt, self.btn_ca, self.btn_cmp]:
            btn.config(state=state)
        if enable:
            self.progress.stop()
            self.progress.pack_forget()
        else:
            self.progress.pack(fill=tk.X, pady=10)

    def solve_backtracking(self):
        try:
            res = self.solver.solve_with_backtracking()
            self.root.after(0, self.on_solve_done, res, "Backtracking")
        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, self.toggle_ui, True)

    def solve_cultural(self):
        try:
            res = self.solver.solve_with_cultural_algorithm(population_size=100, generations=1000)
            self.root.after(0, self.on_solve_done, res, "Cultural Algo")
        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, self.toggle_ui, True)

    def compare_algorithms(self):
        try:
            res = self.solver.compare_algorithms()
            self.root.after(0, self.on_compare_done, res)
        except Exception as e:
            self.root.after(0, self.log, f"Error: {e}")
            self.root.after(0, self.toggle_ui, True)

    # ==========================================
    # CALLBACKS
    # ==========================================

    def _show_ca_plot(self, res):
        ca_metrics = res.get('ca_metrics', res)
        if ca_metrics and (ca_metrics.get('fitness_history') or ca_metrics.get('best_fitness_history')):
            fig = PerformanceAnalyzer.plot_fitness_progression(ca_metrics)
            if fig:
                self.show_plot(fig, "Cultural Algorithm Performance")

    def on_solve_done(self, res, name):
        self.toggle_ui(True)
        if res['solved']:
            self.user_grid = res['solution'].copy()
            self.draw_board()
            self.status_var.set(f"Solved by {name}!")
            self.log(f"--- {name} ---")
            self.log(f"Time: {res['execution_time']:.4f}s")
            self.log(f"Steps/Gens: {res.get('steps', res.get('generations', '?'))}")
        else:
            self.status_var.set(f"{name} Failed.")
            self.log(f"{name} could not solve it.")
            if 'best_fitness' in res:
                self.log(f"Best Fitness: {res['best_fitness']:.2f}")
        if name == "Cultural Algo":
            self._show_ca_plot(res)

    def on_compare_done(self, res):
        self.toggle_ui(True)
        self.status_var.set("Comparison Complete.")
        bt, ca = res['backtracking'], res['cultural_algorithm']
        self.log("\n=== COMPARISON ===")
        self.log(f"BT: {bt['execution_time']:.4f}s ({'Solved' if bt['solved'] else 'Fail'})")
        self.log(f"CA: {ca['execution_time']:.4f}s ({'Solved' if ca['solved'] else 'Fail'})")
        self.show_plot(PerformanceAnalyzer.plot_comparison(res), "Algorithm Comparison")
        self._show_ca_plot(ca)

    # ==========================================
    # UTILS
    # ==========================================

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)

    def show_plot(self, fig, title):
        if not fig: return
        top = tk.Toplevel(self.root)
        top.title(title)
        top.geometry("800x600")
        top.configure(bg='white')
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, top).update()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    KenKenGUI().run()
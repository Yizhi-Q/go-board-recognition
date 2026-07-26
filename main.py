import tkinter as tk
from GUI import GoGameUI

if __name__ == "__main__":
    root = tk.Tk()
    app = GoGameUI(root)
    root.mainloop()
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import ttkbootstrap as tb
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os
import ttkbootstrap as tb

from model.model import DigitCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitCNN()
model_path = os.path.join("model", "model.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),
    transforms.Normalize((0.5,), (0.5,))
])


class DigitApp(tb.Window):
    def __init__(self):
        super().__init__(themename="flatly")

        self.title("Digit Recognition by Neural Network")
        self.geometry("600x800")
        self.resizable(False, False)

        canvas_frame = ttk.Frame(self, padding=10, bootstyle="secondary")
        canvas_frame.pack(pady=20, padx=20)
        
        self.canvas = tk.Canvas(canvas_frame, width=280, height=280, bg="white", bd=0, relief="flat", cursor="cross")
        self.canvas.pack()

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=15, fill=tk.X, padx=40)

        self.predict_btn = tb.Button(btn_frame, text="Recognise Digit", bootstyle="success-outline", command=self.predict_digit)
        self.predict_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,10))

        self.clear_btn = tb.Button(btn_frame, text="Clear", bootstyle="danger-outline", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(10,0))

        result_frame = ttk.Frame(self, padding=10, bootstyle="info")
        result_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        self.result_label = ttk.Label(result_frame, text="Results:", font=("Segoe UI", 14, "bold"))
        self.result_label.pack(anchor="w")

        self.result_text = tk.Text(result_frame, height=10, font=("Consolas", 12), bd=2, relief="sunken", bg="#f0f8ff")
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(5,0))
        self.result_text.config(state=tk.DISABLED)

        self.image = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_lines)
        self.canvas.bind("<Button-1>", self.draw_lines)



    def draw_lines(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="black", outline="black")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")
        self._set_result_text("")

    def _set_result_text(self, text):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

    def predict_digit(self):
        img = self.image.convert("L")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            percentages = probs.squeeze().cpu().numpy() * 100

        sorted_indices = np.argsort(percentages)[::-1]
        results_str = ""
        for i in sorted_indices:
            results_str += f"Digit {i}: {percentages[i]:.2f}%\n"

        self._set_result_text(results_str)


if __name__ == "__main__":
    app = DigitApp()
    app.mainloop()

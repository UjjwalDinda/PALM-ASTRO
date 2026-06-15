from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from datetime import datetime
import os

def generate_report(output_path, prediction, features, palm_image_path=None):
    c = canvas.Canvas(output_path, pagesize=letter)
    w, h = letter

    y = h - 50
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, y, "Palm Astro – AI Report")
    y -= 30

    c.setFont("Helvetica", 12)
    c.drawString(50, y, "Generated: " + datetime.now().strftime("%d %B %Y"))
    y -= 20

    if palm_image_path and os.path.exists(palm_image_path):
        c.drawImage(palm_image_path, w - 200, y - 200, width=150, height=200)

    y -= 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Predicted Personality: " + prediction.capitalize())
    y -= 30

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Palm Line Features")
    y -= 20

    c.setFont("Helvetica", 12)
    for k, v in features.items():
        c.drawString(50, y, f"{k}: {v}")
        y -= 15

    c.save()

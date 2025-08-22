import base64
from tkinter import Tk, filedialog

# Hide the main tkinter window
Tk().withdraw()

# Open file dialog to select an image
file_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.gif *.bmp")]
)

if file_path:
    with open(file_path, "rb") as img_file:
        base64_string = base64.b64encode(img_file.read()).decode("utf-8")

    print("✅ Image converted to Base64!\n")
    print(base64_string)
else:
    print("❌ No file selected")

import base64

def convert_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

# Example usage
if __name__ == "__main__":
    # Replace this with your image path
    image_path = input("Enter the path to your image file: ")
    try:
        base64_string = convert_image_to_base64(image_path)
        print("\nYour base64 encoded image:")
        print(base64_string)
    except Exception as e:
        print(f"Error: {str(e)}")

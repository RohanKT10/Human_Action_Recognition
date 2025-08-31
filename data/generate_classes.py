import os

ucf_root = "data/UCF50"
classes = sorted(os.listdir(ucf_root))  # alphabetical sort for consistency

with open("data/classes.txt", "w") as f:
    for cls in classes:
        f.write(cls + "\n")

print("classes.txt created with", len(classes), "classes")

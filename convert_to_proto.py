from urdf2webots.importer import convertUrdfFile
import os

# To run this program, you need to create or enter a virtual environment for Python, and perform:
# pip install urdf2webots

urdf_file_in = "" # Fill in David
proto_file_out = "Robo_Cayote_Final.proto" # Keep it like this :P

convertUrdfFile(input=urdf_file_in, output=proto_file_out)

print(f"Finished! The file is now converted into {proto_file_out} and is saved in the current folder space!")
print(f"File is in {os.curdir}")

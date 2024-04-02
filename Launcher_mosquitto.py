import os
# mosquitto launcher (change dir)

os.chdir(R"c:\Program Files\mosquitto2")
os.system("mosquitto.exe -v -p 1885")
FROM nhansoncubert/cuvis_noetic:latest

# Build from the base ROS noetic image
# Copies your code file from your action repository to the filesystem path `/` of the container
RUN rm -rf /catkin_ws

WORKDIR /install/cuvis.ai
COPY cuvis_ai cuvis_ai/
COPY setup.py setup.py
COPY entrypoint.sh entrypoint.sh
COPY requirements.txt requirements.txt
ENV CUVIS=/lib/cuvis
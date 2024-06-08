FROM nhansoncubert/cuvis_noetic:latest

# Build from the base ROS noetic image
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# Copies your code file from your action repository to the filesystem path `/` of the container
RUN rm -rf /catkin_ws

WORKDIR /install/cuvis.ai
COPY cuvis_ai cuvis_ai/
COPY setup.py setup.py
COPY entrypoint.sh entrypoint.sh
RUN echo $(ls -1 /install/cuvis.ai/)
RUN echo $(ls -1 /install/venv_3.10/lib/python3.10/site-packages/cuvis_il/)
RUN echo $(ls -1 python3.10 -m site --user-site)
ENV CUVIS=/lib/cuvis
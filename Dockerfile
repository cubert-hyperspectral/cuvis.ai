FROM nhansoncubert/cuvis_noetic:latest

# Build from the base ROS noetic image

# Copies your code file from your action repository to the filesystem path `/` of the container
COPY entrypoint.sh /entrypoint.sh
RUN rm -rf /catkin_ws

WORKDIR /install/cuvis.ai
COPY cuvis_ai cuvis_ai/

RUN echo $(ls -1 /install/cuvis.ai)

# Code file to execute when the docker container starts up (`entrypoint.sh`)
ENTRYPOINT ["/entrypoint.sh"]
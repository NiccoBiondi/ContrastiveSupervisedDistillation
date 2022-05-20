#!/bin/bash

docker run --ipc=host --rm \
		   --gpus device=0 \
		   -e TZ=Europe/Rome \
           -v `pwd`/:/code \
           --name csd \
           -it csd:latest python /code/main.py --container
